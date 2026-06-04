"""
on_prem_oss_unimodel.py
═══════════════════════════════════════════════════════════════════════════════
NGI Pharma Voice AI — Single-model, multi-KV-cache concurrent agent demo.

Architecture
────────────
  ONE vLLM server  (single model instance, one weight copy in VRAM)
  FIVE agent threads  (each with its own independent conversation history /
                       KV-cache context — vLLM allocates a fresh sequence slot
                       per request, so sessions are fully isolated)

  All five agents POST to the same HTTP endpoint concurrently.
  vLLM's continuous batching merges the requests into shared GPU batches,
  routing them to the single model. Each agent's KV cache grows independently.

Latency measurements
────────────────────
  resolver_ms   — pure in-process DB lookup time (μs-range baseline)
  ttft_ms       — time-to-first-token  (dispatch → first streaming chunk)
                  captures queue wait + prefill time
  generation_ms — time from first token to last token (decode phase only)
  e2e_ms        — full round-trip: request dispatch → complete response
                  includes queue wait, prefill, decode
  routing_ms    — e2e_ms − resolver_ms  (everything the model + server added)
  hiccup        — flagged when ttft_ms > 3× median TTFT for that scenario,
                  indicating a scheduling/batching stall

GPU allocation (single server, TP only — no DP needed)
──────────────────────────────────────────────────────
  Model               dtype    TP   GPUs
  ──────────────────────────────────────
  GPT OSS 120B MXFP4  mxfp4    1     1   (60 GB fits in 1× RTX PRO 6000)
  GPT OSS 120B BF16   bf16     4     4   (240 GB → 4× RTX PRO 6000 / 2× B200)
  Gemma 4 26B  BF16   bf16     1     1
  Gemma 4 31B  BF16   bf16     1     1
  Qwen3 72B    FP8    fp8      1     1
  Qwen3 72B    BF16   bf16     2     2

Usage
─────
    modal run on_prem_oss_unimodel.py
    modal run on_prem_oss_unimodel.py --model gemma4_26b
    modal run on_prem_oss_unimodel.py --model qwen3_72b_fp8 --scenario 2
    modal run on_prem_oss_unimodel.py --scenario 3

    ACTIVE_GPU=B200        modal run on_prem_oss_unimodel.py
    ACTIVE_GPU=RTX_PRO_6000 modal run on_prem_oss_unimodel.py --model gemma4_26b
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, median, stdev
from typing import Any

import modal

# ─────────────────────────────────────────────────────────────────────────────
# Runtime configuration
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_GPU: str = os.environ.get("ACTIVE_GPU", "RTX_PRO_6000")
N_AGENTS:   int = int(os.environ.get("N_AGENTS", "5"))
VLLM_PORT:  int = 8100

assert ACTIVE_GPU in ("RTX_PRO_6000", "B200"), (
    f"Unknown ACTIVE_GPU={ACTIVE_GPU!r}. Choose 'RTX_PRO_6000' or 'B200'."
)

_MODAL_GPU_TAG: dict[str, str] = {
    "RTX_PRO_6000": "RTX-PRO-6000",
    "B200":         "B200",
}

# HICCUP_MULTIPLIER: a request whose TTFT exceeds this multiple of the scenario
# median TTFT is flagged as a scheduling hiccup.
HICCUP_MULTIPLIER: float = float(os.environ.get("HICCUP_MULTIPLIER", "3.0"))


# ─────────────────────────────────────────────────────────────────────────────
# Modal app + image
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("pharma-single-model-demo")

if ACTIVE_GPU == "B200":
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04",
            add_python="3.11",
        )
        .pip_install(
            "vllm>=0.6.0",
            "openai>=1.30.0",
            "huggingface-hub>=0.23.0",
        )
        .env({"VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    )
elif ACTIVE_GPU == "RTX_PRO_6000":
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-devel-ubuntu22.04",
            add_python="3.11",
        )
        .pip_install(
            "vllm>=0.6.0",
            "openai>=1.30.0",
            "huggingface-hub>=0.23.0",
        )
        .env({"VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    )

model_volume   = modal.Volume.from_name("pharma-model-weights", create_if_missing=True)
MODEL_CACHE    = "/model-cache"
results_volume = modal.Volume.from_name("pharma-results", create_if_missing=True)
RESULTS_DIR    = "/results"


# ─────────────────────────────────────────────────────────────────────────────
# GPU truth table  (TP per GPU target — NO DP, single server)
# ─────────────────────────────────────────────────────────────────────────────

# Values are tensor_parallel (= GPU count) per hardware target.
_TP_TABLE: dict[str, dict[str, int]] = {
    #                          RTX_PRO_6000  B200
    "gpt120b_mxfp4":  {"RTX_PRO_6000": 1, "B200": 1},
    "gpt120b_bf16":   {"RTX_PRO_6000": 1, "B200": 1},
    "gemma4_26b":     {"RTX_PRO_6000": 1, "B200": 1},
    "gemma4_31b":     {"RTX_PRO_6000": 1, "B200": 1},
    "qwen3_72b_fp8":  {"RTX_PRO_6000": 1, "B200": 1},
    "qwen3_72b_bf16": {"RTX_PRO_6000": 1, "B200": 1},
}


def _tp(key: str) -> int:
    return _TP_TABLE[key][ACTIVE_GPU]


def _modal_gpu(key: str) -> str:
    n   = _tp(key)
    tag = _MODAL_GPU_TAG[ACTIVE_GPU]
    return f"{tag}:{n}" if n > 1 else tag


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    key: str
    hf_repo: str
    display_name: str
    dtype: str
    quantization: str | None
    tensor_parallel: int      # = GPU count (TP only, no DP)
    max_model_len: int
    tool_mode: str            # "native" | "json"
    tool_call_parser: str = "hermes"
    extra_vllm_args: list[str] = field(default_factory=list)

    @property
    def gpu_count(self) -> int:
        return self.tensor_parallel


def _cfg(
    key: str,
    hf_repo: str,
    dtype: str,
    quantization: str | None,
    max_model_len: int,
    tool_mode: str,
    tool_call_parser: str = "hermes",
    extra: list[str] | None = None,
) -> ModelConfig:
    tp  = _tp(key)
    tag = _MODAL_GPU_TAG[ACTIVE_GPU]
    q   = f" {quantization.upper()}" if quantization else " BF16"
    return ModelConfig(
        key=key, hf_repo=hf_repo,
        display_name=f"{key}{q} — {tp}× {tag} (TP={tp} DP=1 single-server)",
        dtype=dtype, quantization=quantization,
        tensor_parallel=tp, max_model_len=max_model_len,
        tool_mode=tool_mode, tool_call_parser=tool_call_parser,
        extra_vllm_args=extra or [],
    )


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt120b_mxfp4": _cfg(
        "gpt120b_mxfp4", "openai/gpt-oss-120b",
        "bfloat16", "mxfp4", 32768, "json",
        extra=["--enable-chunked-prefill"],
    ),
    "gpt120b_bf16": _cfg(
        "gpt120b_bf16", "openai/gpt-oss-120b",
        "bfloat16", None, 32768, "json",
        extra=["--enable-chunked-prefill"],
    ),
    "gemma4_26b": _cfg(
        "gemma4_26b", "google/gemma-3-27b-it",
        "bfloat16", None, 32768, "json",
    ),
    "gemma4_31b": _cfg(
        "gemma4_31b", "google/gemma-3-27b-it",
        "bfloat16", None, 32768, "json",
    ),
    "qwen3_72b_fp8": _cfg(
        "qwen3_72b_fp8", "Qwen/Qwen2.5-72B-Instruct",
        "bfloat16", "fp8", 32768, "native",
    ),
    "qwen3_72b_bf16": _cfg(
        "qwen3_72b_bf16", "Qwen/Qwen2.5-72B-Instruct",
        "bfloat16", None, 32768, "native",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# FAKE IN-MEMORY DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

_DB: dict[str, Any] = {
    "members": {
        "784-1996-7169603-3": {
            "emirates_id":           "784-1996-7169603-3",
            "name":                  "Omar Ali",
            "dob":                   "1996-05-15",
            "policy_number":         "ADNIC-ENH-001",
            "insurer":               "ADNIC",
            "plan":                  "ADNIC Enhanced",
            "plan_tier":             "Mid",
            "status":                "active",
            "copay_pct":             20,
            "annual_limit_aed":      300_000,
            "remaining_benefit_aed": 241_500,
            "network_pharmacy":      "DXB-PH-005",
            "expiry_date":           None,
            "policy_start_date":     "2024-06-01",
            "policy_end_date":       "2025-05-31",
        },
        "784-2004-2137407-6": {
            "emirates_id":           "784-2004-2137407-6",
            "name":                  "Ahmed Khan",
            "dob":                   "1988-03-16",
            "policy_number":         "NAS-ENH-042",
            "insurer":               "NAS",
            "plan":                  "NAS Enhanced",
            "plan_tier":             "Mid",
            "status":                "active",
            "copay_pct":             10,
            "annual_limit_aed":      300_000,
            "remaining_benefit_aed": 287_400,
            "network_pharmacy":      None,
            "expiry_date":           None,
            "policy_start_date":     "2024-04-15",
            "policy_end_date":       "2025-04-14",
        },
        "784-1983-4821093-1": {
            "emirates_id":           "784-1983-4821093-1",
            "name":                  "Hana Patel",
            "dob":                   "1993-04-21",
            "policy_number":         "ADNIC-ENH-077",
            "insurer":               "ADNIC",
            "plan":                  "ADNIC Enhanced",
            "plan_tier":             "Mid",
            "status":                "active",
            "copay_pct":             10,
            "annual_limit_aed":      300_000,
            "remaining_benefit_aed": 280_856,
            "network_pharmacy":      "DXB-PH-022",
            "expiry_date":           None,
            "policy_start_date":     "2024-11-06",
            "policy_end_date":       "2025-11-05",
        },
        "784-1978-6329401-7": {
            "emirates_id":           "784-1978-6329401-7",
            "name":                  "Ravi Reyes",
            "dob":                   "1978-02-08",
            "policy_number":         "DAMAN-GLD-199",
            "insurer":               "Daman",
            "plan":                  "Daman Gold",
            "plan_tier":             "High",
            "status":                "active",
            "copay_pct":             5,
            "annual_limit_aed":      400_000,
            "remaining_benefit_aed": 142_163,
            "network_pharmacy":      "DXB-PH-005",
            "expiry_date":           None,
            "policy_start_date":     "2024-11-20",
            "policy_end_date":       "2025-11-19",
        },
        "784-1985-7741823-5": {
            "emirates_id":           "784-1985-7741823-5",
            "name":                  "Deepa Ali",
            "dob":                   "1985-01-20",
            "policy_number":         "AXA-BSC-304",
            "insurer":               "AXA Gulf",
            "plan":                  "AXA Basic",
            "plan_tier":             "Basic",
            "status":                "active",
            "copay_pct":             20,
            "annual_limit_aed":      150_000,
            "remaining_benefit_aed": 45_921,
            "network_pharmacy":      "DXB-PH-029",
            "expiry_date":           None,
            "policy_start_date":     "2024-12-05",
            "policy_end_date":       "2025-12-04",
        },
        "784-1983-5524190-4": {
            "emirates_id":           "784-1983-5524190-4",
            "name":                  "Nadia Ibrahim",
            "dob":                   "2002-02-26",
            "policy_number":         "DAMAN-THQ-088",
            "insurer":               "Daman",
            "plan":                  "Daman Thiqa",
            "plan_tier":             "Premium",
            "status":                "active",
            "copay_pct":             0,
            "annual_limit_aed":      500_000,
            "remaining_benefit_aed": 291_328,
            "network_pharmacy":      "DXB-PH-029",
            "expiry_date":           None,
            "policy_start_date":     "2024-09-27",
            "policy_end_date":       "2025-09-26",
        },
        "784-1974-3341057-2": {
            "emirates_id":           "784-1974-3341057-2",
            "name":                  "Fatima Al Mansoori",
            "dob":                   "1982-05-05",
            "policy_number":         "CIGNA-ME-117",
            "insurer":               "Cigna ME",
            "plan":                  "Cigna ME Standard",
            "plan_tier":             "Basic",
            "status":                "expired",
            "copay_pct":             0,
            "annual_limit_aed":      150_000,
            "remaining_benefit_aed": 0,
            "network_pharmacy":      None,
            "expiry_date":           "2024-12-16",
            "policy_start_date":     "2023-12-17",
            "policy_end_date":       "2024-12-16",
        },
    },
    "claims": [
        {
            "claim_id": "CLM-2025-0441", "member_id": "784-1996-7169603-3",
            "drug": "Zocor 40mg", "generic": "Simvastatin 40mg", "drug_class": "statin",
            "status": "under_review", "pa_required": True,
            "pa_reason": (
                "Step therapy applies — documentation of prior failed therapy "
                "with Simvastatin or Lovastatin required before this brand is approved."
            ),
            "rejection_reason": None, "submitted": "2025-05-20",
        },
        {
            "claim_id": "CLM-2025-0512", "member_id": "784-2004-2137407-6",
            "drug": "Januvia 100mg", "generic": "Sitagliptin 100mg", "drug_class": "DPP-4 inhibitor",
            "status": "under_review", "pa_required": True,
            "pa_reason": (
                "Prior Authorization required per NAS formulary Tier 3 policy. "
                "Physician must submit PA form with clinical notes via E-Claim portal."
            ),
            "rejection_reason": None, "submitted": "2025-05-22",
        },
        {
            "claim_id": "CLM-2025-0490", "member_id": "784-2004-2137407-6",
            "drug": "Metformin 500mg", "generic": "Metformin 500mg", "drug_class": "biguanide",
            "status": "approved", "pa_required": False,
            "pa_reason": None, "rejection_reason": None, "submitted": "2025-05-10",
        },
        {
            "claim_id": "CLM-2025-0530", "member_id": "784-1974-3341057-2",
            "drug": "Plavix", "generic": "Clopidogrel 75mg", "drug_class": "antiplatelet",
            "status": "rejected", "pa_required": False, "pa_reason": None,
            "rejection_reason": "Policy expired on 2024-12-16; no active coverage.",
            "submitted": "2025-05-23",
        },
        {
            "claim_id": "CLM-2025-0601", "member_id": "784-1983-4821093-1",
            "drug": "Lantus", "generic": "Insulin Glargine", "drug_class": "insulin",
            "status": "under_review", "pa_required": True,
            "pa_reason": (
                "Insulin Glargine (Lantus) requires PA under ADNIC Enhanced plan. "
                "Physician must submit clinical justification confirming HbA1c > 8.5%."
            ),
            "rejection_reason": None, "submitted": "2025-05-24",
        },
        {
            "claim_id": "CLM-2025-0617", "member_id": "784-1978-6329401-7",
            "drug": "Zocor 20mg", "generic": "Simvastatin 20mg", "drug_class": "statin",
            "status": "rejected", "pa_required": False, "pa_reason": None,
            "rejection_reason": (
                "Brand Zocor restricted to generic list under Daman Gold. "
                "Please resubmit with Simvastatin 20mg (generic)."
            ),
            "submitted": "2025-05-21",
        },
        {
            "claim_id": "CLM-2025-0633", "member_id": "784-1985-7741823-5",
            "drug": "Nexium 40mg", "generic": "Esomeprazole 40mg", "drug_class": "PPI",
            "status": "rejected", "pa_required": False, "pa_reason": None,
            "rejection_reason": (
                "Esomeprazole 80mg dose not covered under AXA Basic. "
                "Standard 40mg covered; please adjust prescription."
            ),
            "submitted": "2025-05-25",
        },
        {
            "claim_id": "CLM-2025-0655", "member_id": "784-1983-5524190-4",
            "drug": "Amoxil 500mg", "generic": "Amoxicillin 500mg", "drug_class": "antibiotic",
            "status": "approved", "pa_required": False,
            "pa_reason": None, "rejection_reason": None, "submitted": "2025-05-26",
        },
        {
            "claim_id": "CLM-2025-0671", "member_id": "784-1983-5524190-4",
            "drug": "Lantus", "generic": "Insulin Glargine", "drug_class": "insulin",
            "status": "under_review", "pa_required": True,
            "pa_reason": (
                "Insulin Glargine requires PA under Daman Thiqa plan for new patients. "
                "Submit HbA1c readings and endocrinologist letter via E-Claim."
            ),
            "rejection_reason": None, "submitted": "2025-05-27",
        },
    ],
    "inventory": {
        "DXB-PH-005": {
            "Atorvastatin 20mg": {"qty": 240, "status": "in_stock"},
            "Rosuvastatin 10mg": {"qty": 18,  "status": "low_stock"},
            "Simvastatin 20mg":  {"qty": 75,  "status": "in_stock"},
            "Zocor 40mg":        {"qty": 45,  "status": "in_stock"},
            "Januvia 100mg":     {"qty": 90,  "status": "in_stock"},
            "Metformin 500mg":   {"qty": 300, "status": "in_stock"},
            "Plavix":            {"qty": 60,  "status": "in_stock"},
            "Aspirin 81mg":      {"qty": 500, "status": "in_stock"},
            "Lantus":            {"qty": 12,  "status": "low_stock"},
            "Insulin Detemir":   {"qty": 30,  "status": "in_stock"},
        },
        "DXB-PH-022": {
            "Atorvastatin 20mg":  {"qty": 180, "status": "in_stock"},
            "Rosuvastatin 10mg":  {"qty": 0,   "status": "out_of_stock"},
            "Amoxil 500mg":       {"qty": 120, "status": "in_stock"},
            "Azithromycin 250mg": {"qty": 60,  "status": "in_stock"},
            "Metformin 500mg":    {"qty": 200, "status": "in_stock"},
            "Lantus":             {"qty": 0,   "status": "out_of_stock"},
            "Insulin Detemir":    {"qty": 20,  "status": "in_stock"},
        },
        "DXB-PH-029": {
            "Atorvastatin 20mg": {"qty": 95,  "status": "in_stock"},
            "Simvastatin 20mg":  {"qty": 50,  "status": "in_stock"},
            "Nexium 40mg":       {"qty": 0,   "status": "out_of_stock"},
            "Omeprazole 20mg":   {"qty": 150, "status": "in_stock"},
            "Pantoprazole 40mg": {"qty": 8,   "status": "low_stock"},
            "Metformin 500mg":   {"qty": 400, "status": "in_stock"},
            "Aspirin 81mg":      {"qty": 350, "status": "in_stock"},
            "Amoxil 500mg":      {"qty": 80,  "status": "in_stock"},
            "Lantus":            {"qty": 25,  "status": "in_stock"},
        },
    },
    "formulary_alternatives": {
        "statin": [
            {"drug": "Atorvastatin 20mg", "tier": 1, "covered": True,  "pa_required": False},
            {"drug": "Rosuvastatin 10mg",  "tier": 2, "covered": True,  "pa_required": False},
            {"drug": "Simvastatin 20mg",   "tier": 1, "covered": True,  "pa_required": False},
        ],
        "DPP-4 inhibitor": [
            {"drug": "Metformin 500mg",    "tier": 1, "covered": True,  "pa_required": False},
            {"drug": "Januvia 100mg",      "tier": 3, "covered": True,  "pa_required": True},
        ],
        "biguanide": [
            {"drug": "Metformin 500mg",    "tier": 1, "covered": True,  "pa_required": False},
        ],
        "antiplatelet": [
            {"drug": "Aspirin 81mg",       "tier": 1, "covered": True,  "pa_required": False},
            {"drug": "Ticagrelor 90mg",    "tier": 3, "covered": True,  "pa_required": True},
        ],
        "insulin": [
            {"drug": "Insulin Detemir",    "tier": 2, "covered": True,  "pa_required": False},
            {"drug": "NPH Insulin",        "tier": 1, "covered": True,  "pa_required": False},
        ],
        "antibiotic": [
            {"drug": "Azithromycin 250mg",   "tier": 2, "covered": True, "pa_required": False},
            {"drug": "Clarithromycin 500mg", "tier": 2, "covered": True, "pa_required": False},
        ],
        "PPI": [
            {"drug": "Omeprazole 20mg",    "tier": 1, "covered": True, "pa_required": False},
            {"drug": "Pantoprazole 40mg",  "tier": 1, "covered": True, "pa_required": False},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS_OPENAI: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_member",
            "description": (
                "Look up an insurance member by Emirates ID. Returns policy metadata. "
                "Does NOT return member name — verify via verify_member_name first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emirates_id": {"type": "string", "description": "e.g. 784-1996-7169603-3"},
                },
                "required": ["emirates_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_member_name",
            "description": (
                "Verify caller's stated name matches the record. "
                "Call AFTER lookup_member. Returns {verified: true/false}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emirates_id":   {"type": "string"},
                    "provided_name": {"type": "string"},
                },
                "required": ["emirates_id", "provided_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_claim_status",
            "description": "Retrieve claim status for a drug and verified member.",
            "parameters": {
                "type": "object",
                "properties": {
                    "emirates_id": {"type": "string"},
                    "drug_name":   {"type": "string"},
                },
                "required": ["emirates_id", "drug_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_formulary_alternatives",
            "description": (
                "Get covered formulary alternatives for a drug class with real-time "
                "inventory. Use drug_class from get_claim_status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_class":  {"type": "string"},
                    "pharmacy_id": {"type": "string"},
                },
                "required": ["drug_class", "pharmacy_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_policy_status",
            "description": "Check if a member's policy is active or expired.",
            "parameters": {
                "type": "object",
                "properties": {
                    "emirates_id": {"type": "string"},
                },
                "required": ["emirates_id"],
            },
        },
    },
]

_TOOLS_JSON_SCHEMA = json.dumps(
    [t["function"] for t in TOOLS_OPENAI], indent=2, ensure_ascii=False
)

_JSON_SHIM_ADDENDUM = f"""
You have access to the following tools. When you need to call a tool, output
ONLY a valid JSON object on a single line — no prose, no markdown fences:

  {{"tool": "<tool_name>", "arguments": {{...}}}}

After receiving the tool result (injected as a user message), continue naturally
in plain text. If no tool call is needed, respond normally in plain text.

Available tools:
{_TOOLS_JSON_SCHEMA}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

def execute_tool(name: str, inputs: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()

    if name == "lookup_member":
        eid = inputs["emirates_id"].strip()
        m   = _DB["members"].get(eid)
        result: dict = (
            {"found": False, "error": "No member record."}
            if not m else {
                "found": True,
                "emirates_id": m["emirates_id"], "policy_number": m["policy_number"],
                "insurer": m["insurer"], "plan": m["plan"], "status": m["status"],
                "copay_pct": m["copay_pct"], "expiry_date": m["expiry_date"],
                "network_pharmacy": m["network_pharmacy"],
            }
        )

    elif name == "verify_member_name":
        eid      = inputs["emirates_id"].strip()
        provided = inputs["provided_name"].strip().lower()
        m        = _DB["members"].get(eid)
        if not m:
            result = {"verified": False, "reason": "Member not found."}
        else:
            stored = m["name"].strip().lower()
            result = {"verified": (provided == stored) or (provided in stored) or (stored in provided)}

    elif name == "get_claim_status":
        eid   = inputs["emirates_id"].strip()
        query = inputs["drug_name"].strip().lower()
        match = next(
            (c for c in _DB["claims"]
             if c["member_id"] == eid and (
                 query in c["drug"].lower() or query in c["generic"].lower()
                 or c["drug"].lower() in query
             )),
            None,
        )
        result = (
            {"found": False, "message": "No claim found."} if not match else
            {
                "found": True, "claim_id": match["claim_id"],
                "drug": match["drug"], "generic": match["generic"],
                "drug_class": match["drug_class"], "status": match["status"],
                "pa_required": match["pa_required"], "pa_reason": match["pa_reason"],
                "rejection_reason": match["rejection_reason"], "submitted": match["submitted"],
            }
        )

    elif name == "get_formulary_alternatives":
        dc   = inputs["drug_class"].strip().lower()
        pid  = inputs["pharmacy_id"].strip()
        alts = _DB["formulary_alternatives"].get(dc, [])
        inv  = _DB["inventory"].get(pid, {})
        result = {
            "drug_class": dc, "pharmacy_id": pid,
            "alternatives": [
                {**a,
                 "inventory_status": inv.get(a["drug"], {}).get("status", "unknown"),
                 "qty_on_hand":      inv.get(a["drug"], {}).get("qty", 0)}
                for a in alts
            ],
        }

    elif name == "get_policy_status":
        eid = inputs["emirates_id"].strip()
        m   = _DB["members"].get(eid)
        result = (
            {"found": False} if not m else
            {
                "found": True, "policy_number": m["policy_number"],
                "insurer": m["insurer"], "plan": m["plan"],
                "status": m["status"], "expiry_date": m["expiry_date"],
            }
        )

    else:
        result = {"error": f"Unknown tool: {name}"}

    return result, (time.perf_counter() - t0) * 1_000


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_BASE = """\
You are the NGI Pharma AI Agent — an autonomous voice agent handling inbound calls
for a Pharmacy Benefit Management (PBM) platform operated by IIRIS Health.

IDENTITY & VERIFICATION RULES
1. Authenticate before disclosing any protected information.
   - Pharmacy caller: ask for pharmacy branch ID + patient Emirates ID, then confirm name.
   - Patient caller: ask for Emirates ID and date of birth, then confirm full name.
2. Call lookup_member first, then verify_member_name. Disclose nothing until
   verify_member_name returns {verified: true}.
3. If verification fails: "I'm unable to verify the identity on record." Do NOT reveal stored name.

CLAIM & POLICY RULES
4. Use get_claim_status with the exact drug the caller mentions.
5. When a claim is "under_review" due to PA, explain why, what must be submitted,
   and that review takes 24-48 hours.
6. When suggesting alternatives, include inventory from get_formulary_alternatives.
7. If a policy is "expired", direct the caller to HR or insurer. Do not process claims.
8. Use get_policy_status as a fast-path when a claim is rejected.

VOICE BEHAVIOR
- Phone call: 2-4 sentences per turn. No bullets or headers.
- Professional, warm, efficient.
- Always use tools. Never invent data.
"""

def build_system_prompt(tool_mode: str) -> str:
    return _SYSTEM_BASE + ("\n" + _JSON_SHIM_ADDENDUM if tool_mode == "json" else "")


# ═══════════════════════════════════════════════════════════════════════════════
# JSON-SHIM PARSER
# ═══════════════════════════════════════════════════════════════════════════════

_JSON_TOOL_RE  = re.compile(
    r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', re.DOTALL
)
_GEMMA_TOOL_RE = re.compile(
    r'```(?:tool_code|python)?\s*\n(\w+)\(([^)]*)\)\s*\n```', re.DOTALL
)

def _parse_gemma(text: str) -> dict | None:
    m = _GEMMA_TOOL_RE.search(text)
    if not m:
        return None
    args: dict = {}
    for kv in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\S+?)(?:,|$))', m.group(2)):
        args[kv.group(1)] = kv.group(2) or kv.group(3) or kv.group(4)
    return {"name": m.group(1), "arguments": args} if m.group(1) and args else None

def parse_json_tool_call(text: str) -> dict | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool" in obj and "arguments" in obj:
            return {"name": obj["tool"], "arguments": obj["arguments"]}
    except json.JSONDecodeError:
        pass
    m = _JSON_TOOL_RE.search(text)
    if m:
        try:
            return {"name": m.group(1), "arguments": json.loads(m.group(2))}
        except json.JSONDecodeError:
            pass
    return _parse_gemma(text)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS  — per LLM call, tracking timing at the HTTP streaming level
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMCallMetrics:
    """Timing for one streaming chat-completion call."""
    agent_id:      int
    turn:          int
    call_index:    int       # nth LLM call within this turn (tool loops)
    dispatch_ts:   float     # time.perf_counter() just before POST
    ttft_ms:       float     # time from dispatch to first token chunk
    generation_ms: float     # time from first token to last token (decode)
    e2e_ms:        float     # dispatch to full response received
    tokens_out:    int       # completion tokens
    hiccup:        bool = False   # set by post-processing


@dataclass
class ToolCallMetrics:
    agent_id:    int
    turn:        int
    tool_name:   str
    resolver_ms: float   # pure DB lookup
    e2e_ms:      float   # LLM dispatch → resolver done (includes queue + prefill + decode)
    routing_ms:  float   # e2e_ms − resolver_ms
    success:     bool


@dataclass
class AgentMetrics:
    agent_id:          int
    scenario_id:       int
    llm_calls:         list[LLMCallMetrics]   = field(default_factory=list)
    tool_calls:        list[ToolCallMetrics]  = field(default_factory=list)
    tokens_generated:  int   = 0
    scenario_duration_s: float = 0.0
    unresponded_turns: int   = 0

    @property
    def tok_per_s(self) -> float:
        return self.tokens_generated / self.scenario_duration_s if self.scenario_duration_s else 0.0

    @property
    def avg_ttft_ms(self) -> float:
        return mean(c.ttft_ms for c in self.llm_calls) if self.llm_calls else 0.0

    @property
    def avg_e2e_ms(self) -> float:
        return mean(c.e2e_ms for c in self.llm_calls) if self.llm_calls else 0.0

    @property
    def avg_resolver_ms(self) -> float:
        return mean(t.resolver_ms for t in self.tool_calls) if self.tool_calls else 0.0

    @property
    def avg_routing_ms(self) -> float:
        return mean(t.routing_ms for t in self.tool_calls) if self.tool_calls else 0.0

    @property
    def hiccup_count(self) -> int:
        return sum(1 for c in self.llm_calls if c.hiccup)


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING LLM CALL  — captures TTFT, generation time, e2e independently
# ═══════════════════════════════════════════════════════════════════════════════

def _streaming_call(
    client,          # openai.OpenAI
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    tool_choice: str | None,
    agent_id: int,
    turn: int,
    call_index: int,
) -> tuple[str, list | None, LLMCallMetrics]:
    """
    Make a streaming chat-completion call and capture:
      - full response text
      - tool_calls list (native mode) or None
      - LLMCallMetrics with TTFT, generation, e2e timing
    """
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=512,
        stream=True,
        stream_options={"include_usage": True},
    )
    if tools:
        kwargs["tools"]       = tools
        kwargs["tool_choice"] = tool_choice or "auto"

    dispatch_ts = time.perf_counter()
    t_first     = None
    content_buf = []
    tool_calls_raw: list[dict] = []    # accumulate streamed tool_call deltas
    tokens_out  = 0

    stream = client.chat.completions.create(**kwargs)

    for chunk in stream:
        now = time.perf_counter()

        # Track TTFT on the very first content/tool chunk
        if t_first is None:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and (
                (delta.content and delta.content.strip())
                or (hasattr(delta, "tool_calls") and delta.tool_calls)
            ):
                t_first = now

        if not chunk.choices:
            # usage chunk
            if chunk.usage:
                tokens_out = chunk.usage.completion_tokens or 0
            continue

        delta = chunk.choices[0].delta

        # Accumulate text
        if delta.content:
            content_buf.append(delta.content)

        # Accumulate tool_call deltas
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                while len(tool_calls_raw) <= idx:
                    tool_calls_raw.append(
                        {"id": None, "type": "function",
                         "function": {"name": "", "arguments": ""}}
                    )
                if tc_delta.id:
                    tool_calls_raw[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_calls_raw[idx]["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls_raw[idx]["function"]["arguments"] += tc_delta.function.arguments

    t_end = time.perf_counter()

    if t_first is None:
        t_first = dispatch_ts   # nothing streamed back (empty response)

    ttft_ms       = (t_first - dispatch_ts) * 1_000
    generation_ms = (t_end   - t_first)     * 1_000
    e2e_ms        = (t_end   - dispatch_ts) * 1_000

    met = LLMCallMetrics(
        agent_id=agent_id, turn=turn, call_index=call_index,
        dispatch_ts=dispatch_ts,
        ttft_ms=ttft_ms, generation_ms=generation_ms,
        e2e_ms=e2e_ms, tokens_out=tokens_out,
    )

    full_text  = "".join(content_buf)
    tool_calls = tool_calls_raw if tool_calls_raw else None
    return full_text, tool_calls, met


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION RUNNER  (single agent, one independent KV-cache context)
# ═══════════════════════════════════════════════════════════════════════════════

_PRINT_LOCK = threading.Lock()

def run_conversation(
    scenario_name: str,
    caller_turns: list[str],
    caller_type: str,
    cfg: ModelConfig,
    base_url: str,
    agent_id: int,
    scenario_id: int,
) -> AgentMetrics:
    from openai import OpenAI

    client  = OpenAI(base_url=base_url, api_key="not-needed")
    metrics = AgentMetrics(agent_id=agent_id, scenario_id=scenario_id)

    messages: list[dict] = [{"role": "system", "content": build_system_prompt(cfg.tool_mode)}]
    scenario_t0 = time.perf_counter()

    with _PRINT_LOCK:
        print(f"\n  [A{agent_id}] ── {scenario_name} ──  started {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

    for turn_idx, caller_utterance in enumerate(caller_turns, start=1):
        messages.append({"role": "user", "content": caller_utterance})
        responded = False
        call_idx  = 0

        while True:
            call_idx += 1

            # ── NATIVE tool-calling (streaming) ──────────────────────────────
            if cfg.tool_mode == "native":
                t_dispatch = time.perf_counter()
                text, tcs, llm_met = _streaming_call(
                    client, cfg.hf_repo, messages,
                    tools=TOOLS_OPENAI, tool_choice="auto",
                    agent_id=agent_id, turn=turn_idx, call_index=call_idx,
                )
                metrics.llm_calls.append(llm_met)
                metrics.tokens_generated += llm_met.tokens_out

                if text.strip():
                    responded = True
                    with _PRINT_LOCK:
                        print(f"  [A{agent_id}|T{turn_idx}] {text.strip()[:100]}")

                if not tcs:
                    messages.append({"role": "assistant", "content": text})
                    break

                messages.append({
                    "role": "assistant", "content": text,
                    "tool_calls": tcs,
                })

                for tc in tcs:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        fn_args = {}

                    result, resolver_ms = execute_tool(fn_name, fn_args)
                    routing_ms = llm_met.e2e_ms  # full round-trip up to resolver done

                    tc_met = ToolCallMetrics(
                        agent_id=agent_id, turn=turn_idx, tool_name=fn_name,
                        resolver_ms=resolver_ms,
                        e2e_ms=routing_ms + resolver_ms,
                        routing_ms=routing_ms,
                        success="error" not in result,
                    )
                    metrics.tool_calls.append(tc_met)

                    with _PRINT_LOCK:
                        print(
                            f"  [A{agent_id}|T{turn_idx}] TOOL {fn_name}"
                            f"  ttft={llm_met.ttft_ms:.0f}ms"
                            f"  resolve={resolver_ms:.3f}ms"
                            f"  routing={routing_ms:.0f}ms"
                        )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            # ── JSON-SHIM (streaming) ─────────────────────────────────────────
            else:
                t_dispatch = time.perf_counter()
                text, _, llm_met = _streaming_call(
                    client, cfg.hf_repo, messages,
                    tools=None, tool_choice=None,
                    agent_id=agent_id, turn=turn_idx, call_index=call_idx,
                )
                metrics.llm_calls.append(llm_met)
                metrics.tokens_generated += llm_met.tokens_out

                tc = parse_json_tool_call(text)

                if tc is None:
                    if text.strip():
                        responded = True
                        with _PRINT_LOCK:
                            print(f"  [A{agent_id}|T{turn_idx}] {text.strip()[:100]}")
                    messages.append({"role": "assistant", "content": text})
                    break

                fn_name = tc["name"]
                fn_args = tc["arguments"]

                result, resolver_ms = execute_tool(fn_name, fn_args)
                routing_ms = llm_met.e2e_ms

                tc_met = ToolCallMetrics(
                    agent_id=agent_id, turn=turn_idx, tool_name=fn_name,
                    resolver_ms=resolver_ms,
                    e2e_ms=routing_ms + resolver_ms,
                    routing_ms=routing_ms,
                    success="error" not in result,
                )
                metrics.tool_calls.append(tc_met)

                with _PRINT_LOCK:
                    print(
                        f"  [A{agent_id}|T{turn_idx}] TOOL {fn_name}"
                        f"  ttft={llm_met.ttft_ms:.0f}ms"
                        f"  resolve={resolver_ms:.3f}ms"
                        f"  routing={routing_ms:.0f}ms"
                    )

                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": f"[TOOL RESULT for {fn_name}]: {json.dumps(result, ensure_ascii=False)}",
                })

        if not responded:
            metrics.unresponded_turns += 1

    metrics.scenario_duration_s = time.perf_counter() - scenario_t0
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# HICCUP DETECTOR  (post-process after all agents finish a scenario)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_hiccups(all_metrics: list[AgentMetrics]) -> int:
    """
    Flag LLMCallMetrics entries whose TTFT is > HICCUP_MULTIPLIER × median.
    Returns total hiccup count.
    """
    all_ttfts = [c.ttft_ms for m in all_metrics for c in m.llm_calls]
    if not all_ttfts:
        return 0
    med = median(all_ttfts)
    threshold = HICCUP_MULTIPLIER * med
    count = 0
    for m in all_metrics:
        for c in m.llm_calls:
            if c.ttft_ms > threshold:
                c.hiccup = True
                count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

_SCENARIOS: dict[int, dict] = {
    1: {
        "name": "Scenario 1 — Complex Pharmacy Call (Step Therapy, Alternatives & Refill Block)",
        "caller_type": "pharmacy_staff",
        "turns": [
            "Hi, I'm calling from Dubai Pharmacy branch 005. I'd like to check on a claim and a few related queries for two patients.",
            "Sure — first patient: Emirates ID 784-1996-7169603-3.",
            "Yes, the patient's name is Omar Ali.",
            "We submitted a claim for Zocor 40mg about a week ago. Can you tell me the current status?",
            "What exactly does step therapy require in this case?",
            "Understood. What covered statin alternatives are available under his plan, and do you show any inventory for them at this pharmacy?",
            "Good — we'll switch to Atorvastatin 20mg. His copay is 20%, so what would the patient owe per 30-unit dispensing at the unit price on file?",
            "Got it. Once the physician sends the updated script, we re-submit through E-Claim and the review should complete within 24–48 hours, correct?",
            "Now the second query — same member, but for Metformin 500mg. Was there a claim approved for that recently?",
            "Was it already dispensed this cycle? The patient is asking for an early refill.",
            "Okay, I'll let the patient know he needs to wait. Moving on — different patient: Emirates ID 784-1978-6329401-7. Name is Ravi Reyes.",
            "He had a Zocor 20mg claim that was rejected because of the brand restriction. What generic should we resubmit?",
            "Do we have Simvastatin 20mg in stock at DXB-PH-005?",
            "Perfect. We'll resubmit Simvastatin 20mg through E-Claim. Is there anything else needed on the submission form?",
        ],
    },
    2: {
        "name": "Scenario 2 — Patient PA Inquiry, Benefit Check & Family Member Query",
        "caller_type": "patient",
        "turns": [
            "Hello, I'm calling to check the status of my Januvia prescription claim.",
            "My Emirates ID is 784-2004-2137407-6 and my date of birth is March 16th, 1988.",
            "Ahmed Khan.",
            "Yes, that's my prescription. Is it approved?",
            "What does prior authorization involve, and how long does the review take once my doctor submits?",
            "Is there a covered alternative for Januvia that doesn't need PA? My doctor mentioned Metformin.",
            "How much would Metformin cost me with my copay?",
            "Also, I checked my benefit balance online and it shows AED 287,400 remaining — is that correct?",
            "Good. Now a separate question — my mother is also on NAS Enhanced and she's trying to get Insulin Glargine covered. Is that generally covered under NAS Enhanced?",
            "She'd need a prior authorization form. Can you tell me exactly what information to include?",
            "Can I submit it through the E-Claim portal myself, or does it have to go from the physician's office?",
            "Understood. One last thing — I also have an approved Metformin claim from a couple of weeks ago. Has it been dispensed?",
            "Is it too early to refill it?",
            "Okay, I understand. Thank you — I'll call my doctor about the Januvia PA and wait for the Metformin cycle to reset.",
        ],
    },
    3: {
        "name": "Scenario 3 — Expired Policy, Rejection Explanations & Drug Switches",
        "caller_type": "patient",
        "turns": [
            "Hi, I tried to fill my Plavix at the pharmacy but they said the claim was rejected. Can you tell me why?",
            "Sure — my Emirates ID is 784-1974-3341057-2 and my birthday is May 5th, 1982.",
            "Fatima Al Mansoori.",
            "I see. When exactly did my policy expire?",
            "I was not informed. Is there any way to get a temporary override or process it manually for this one prescription?",
            "My company said they renewed it. Could there be a processing delay on your end?",
            "Who do I contact to escalate this if HR says it's renewed but you still show it as expired?",
            "While this gets sorted, is there any out-of-pocket option or a generic alternative for Plavix that would be cheaper?",
            "What dose of Aspirin would substitute for Plavix 75mg, and is it available at my nearest pharmacy?",
            "Different issue — I also submitted a claim for a stomach medication called Nexium 40mg. Was that rejected too?",
            "What was the rejection reason for the Nexium?",
            "Is there a covered alternative for the stomach acid issue that I can get without a prior authorization?",
            "Do you show Pantoprazole in stock at DXB-PH-029?",
            "All right. I'll contact HR today about the policy renewal and pick up Pantoprazole in the meantime. Thank you.",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_stdev(vals: list[float]) -> float:
    return stdev(vals) if len(vals) >= 2 else 0.0

def print_metrics_report(
    cfg: ModelConfig,
    all_metrics: list[AgentMetrics],
    scenario_ids: list[int],
    hiccup_count: int,
) -> None:
    W = 72
    sep  = "═" * W
    sep2 = "─" * W

    print(f"\n{sep}")
    print(f"  SINGLE-MODEL MULTI-KV-CACHE METRICS REPORT")
    print(f"  Model  : {cfg.display_name}")
    print(f"  GPU    : {ACTIVE_GPU}  ({cfg.gpu_count}× GPU, TP={cfg.tensor_parallel}, DP=1)")
    print(f"  Agents : {N_AGENTS} concurrent  (independent KV-cache contexts)")
    print(f"  Hiccup threshold: TTFT > {HICCUP_MULTIPLIER}× median TTFT")
    print(f"{sep}\n")

    all_toks:     list[float] = []
    all_ttft:     list[float] = []
    all_gen:      list[float] = []
    all_e2e_llm:  list[float] = []
    all_resolver: list[float] = []
    all_routing:  list[float] = []

    # ── Per-scenario table ────────────────────────────────────────────────────
    hdr = (f"  {'Scen':<6} {'Ags':>3} {'tok/s':>7} {'TTFT ms':>9} "
           f"{'Gen ms':>8} {'E2E ms':>8} {'Res ms':>9} {'Rout ms':>9} {'Hiccups':>8}")
    print(hdr)
    print(f"  {sep2}")

    for sid in scenario_ids:
        group = [m for m in all_metrics if m.scenario_id == sid]
        if not group:
            continue

        toks     = [m.tok_per_s                           for m in group]
        ttfts    = [c.ttft_ms      for m in group for c in m.llm_calls]
        gens     = [c.generation_ms for m in group for c in m.llm_calls]
        e2es_llm = [c.e2e_ms       for m in group for c in m.llm_calls]
        resolvers= [t.resolver_ms  for m in group for t in m.tool_calls]
        routings = [t.routing_ms   for m in group for t in m.tool_calls]
        hics     = sum(c.hiccup    for m in group for c in m.llm_calls)

        for lst in (toks, ttfts, gens, e2es_llm, resolvers, routings):
            if lst is toks:      all_toks.extend(lst)
            elif lst is ttfts:   all_ttft.extend(lst)
            elif lst is gens:    all_gen.extend(lst)
            elif lst is e2es_llm: all_e2e_llm.extend(lst)
            elif lst is resolvers: all_resolver.extend(lst)
            elif lst is routings:  all_routing.extend(lst)

        print(
            f"  S{sid:<5} {len(group):>3} {mean(toks) if toks else 0:>7.1f}"
            f" {mean(ttfts) if ttfts else 0:>9.1f}"
            f" {mean(gens)  if gens  else 0:>8.1f}"
            f" {mean(e2es_llm) if e2es_llm else 0:>8.1f}"
            f" {mean(resolvers) if resolvers else 0:>9.3f}"
            f" {mean(routings)  if routings  else 0:>9.1f}"
            f" {hics:>8}"
        )

    print(f"  {sep2}\n")

    # ── Summary table (the 4 requested metrics + routing) ────────────────────
    def _row(label: str, vals: list[float], unit: str, fmt: str = ".1f") -> None:
        if not vals:
            print(f"  {label:<46}  {'—':>10}")
            return
        print(
            f"  {label:<46}  {mean(vals):>8{fmt}} {unit}"
            f"  σ={_safe_stdev(vals):.2f}  "
            f"p50={median(vals):.1f}  p95={sorted(vals)[int(len(vals)*0.95)-(1 if len(vals)>1 else 0)]:.1f}"
        )

    print(f"  {'SUMMARY METRIC':<46}  {'MEAN':>10}   σ        p50      p95")
    print(f"  {sep2}")
    _row("1. Avg tok/s per scenario (all agents)", all_toks,     "tok/s")
    _row("2. Avg tok/s all scenarios combined",    all_toks,     "tok/s")
    _row("3. Avg tool resolver latency (DB only)", all_resolver, "ms", ".3f")
    _row("4. Avg tool e2e latency (res+routing)",  all_routing,  "ms")
    print(f"  {sep2}")
    _row("   Time-to-first-token (TTFT)",          all_ttft,     "ms")
    _row("   Generation time (decode phase)",      all_gen,      "ms")
    _row("   LLM call e2e (dispatch→done)",        all_e2e_llm,  "ms")
    _row("   Routing latency (e2e − resolver)",    all_routing,  "ms")

    total_tool_calls = sum(len(m.tool_calls) for m in all_metrics)
    total_ok         = sum(t.success         for m in all_metrics for t in m.tool_calls)
    print(f"\n  {'Total LLM calls (all agents × scenarios)':<46}  {sum(len(m.llm_calls) for m in all_metrics):>10}")
    print(f"  {'Total tool calls':<46}  {total_tool_calls:>10}")
    print(f"  {'  ↳ Successful':<46}  {total_ok:>10}")
    print(f"  {'  ↳ Success rate':<46}  {total_ok/total_tool_calls*100 if total_tool_calls else 0:>9.1f} %")
    print(f"  {'Total scheduling hiccups detected':<46}  {hiccup_count:>10}")
    pct_hic = hiccup_count / len(all_ttft) * 100 if all_ttft else 0.0
    print(f"  {'  ↳ Hiccup rate (% of LLM calls)':<46}  {pct_hic:>9.1f} %")
    unresp = sum(m.unresponded_turns for m in all_metrics)
    print(f"  {'Unresponded turns':<46}  {unresp:>10}")

    print(f"\n{sep}\n")

    # ── Per-agent detail ──────────────────────────────────────────────────────
    print(f"  PER-AGENT DETAIL")
    print(f"  {'Ag':<3} {'S':>2} {'tok/s':>6} {'ttft':>7} {'gen':>7} {'tools':>6} {'res':>8} {'rout':>8} {'hic':>4} {'norsp':>5}")
    print(f"  {sep2}")
    for m in sorted(all_metrics, key=lambda x: (x.scenario_id, x.agent_id)):
        ttfts = [c.ttft_ms for c in m.llm_calls]
        gens  = [c.generation_ms for c in m.llm_calls]
        print(
            f"  A{m.agent_id:<2} {m.scenario_id:>2}"
            f" {m.tok_per_s:>6.1f}"
            f" {mean(ttfts) if ttfts else 0:>7.1f}"
            f" {mean(gens)  if gens  else 0:>7.1f}"
            f" {len(m.tool_calls):>6}"
            f" {m.avg_resolver_ms:>8.3f}"
            f" {m.avg_routing_ms:>8.1f}"
            f" {m.hiccup_count:>4}"
            f" {m.unresponded_turns:>5}"
        )
    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS SERIALISER
# ═══════════════════════════════════════════════════════════════════════════════

def _dump_results(
    cfg: ModelConfig,
    all_metrics: list[AgentMetrics],
    scenario_ids: list[int],
    hiccup_count: int,
    run_ts: str,
) -> None:
    all_toks    = [m.tok_per_s                             for m in all_metrics]
    all_ttft    = [c.ttft_ms      for m in all_metrics     for c in m.llm_calls]
    all_gen     = [c.generation_ms for m in all_metrics    for c in m.llm_calls]
    all_e2e_llm = [c.e2e_ms       for m in all_metrics     for c in m.llm_calls]
    all_resolver= [t.resolver_ms  for m in all_metrics     for t in m.tool_calls]
    all_routing = [t.routing_ms   for m in all_metrics     for t in m.tool_calls]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": 0, "stdev": 0, "p50": 0, "p95": 0, "n": 0}
        sv = sorted(vals)
        return {
            "mean":  round(mean(vals), 4),
            "stdev": round(_safe_stdev(vals), 4),
            "p50":   round(median(vals), 4),
            "p95":   round(sv[int(len(sv) * 0.95) - (1 if len(sv) > 1 else 0)], 4),
            "n":     len(vals),
        }

    payload = {
        "run_timestamp":    run_ts,
        "active_gpu":       ACTIVE_GPU,
        "model_key":        cfg.key,
        "display_name":     cfg.display_name,
        "hf_repo":          cfg.hf_repo,
        "gpu_count":        cfg.gpu_count,
        "tensor_parallel":  cfg.tensor_parallel,
        "data_parallel":    1,
        "dtype":            cfg.dtype,
        "quantization":     cfg.quantization,
        "tool_mode":        cfg.tool_mode,
        "n_agents":         N_AGENTS,
        "hiccup_multiplier":HICCUP_MULTIPLIER,
        "summary": {
            "tok_per_s":            _stats(all_toks),
            "ttft_ms":              _stats(all_ttft),
            "generation_ms":        _stats(all_gen),
            "llm_e2e_ms":           _stats(all_e2e_llm),
            "tool_resolver_ms":     _stats(all_resolver),
            "tool_routing_ms":      _stats(all_routing),
            "total_llm_calls":      len(all_ttft),
            "total_tool_calls":     sum(len(m.tool_calls) for m in all_metrics),
            "tool_calls_ok":        sum(t.success for m in all_metrics for t in m.tool_calls),
            "hiccup_count":         hiccup_count,
            "hiccup_rate_pct":      round(hiccup_count / len(all_ttft) * 100, 2) if all_ttft else 0.0,
            "unresponded_turns":    sum(m.unresponded_turns for m in all_metrics),
        },
        "per_agent": [
            {
                "agent_id":           m.agent_id,
                "scenario_id":        m.scenario_id,
                "tok_per_s":          round(m.tok_per_s, 3),
                "tokens_generated":   m.tokens_generated,
                "scenario_duration_s":round(m.scenario_duration_s, 3),
                "avg_ttft_ms":        round(m.avg_ttft_ms, 2),
                "avg_e2e_ms":         round(m.avg_e2e_ms, 2),
                "avg_resolver_ms":    round(m.avg_resolver_ms, 4),
                "avg_routing_ms":     round(m.avg_routing_ms, 2),
                "hiccup_count":       m.hiccup_count,
                "unresponded_turns":  m.unresponded_turns,
                "llm_calls": [
                    {
                        "turn": c.turn, "call_index": c.call_index,
                        "ttft_ms": round(c.ttft_ms, 2),
                        "generation_ms": round(c.generation_ms, 2),
                        "e2e_ms": round(c.e2e_ms, 2),
                        "tokens_out": c.tokens_out,
                        "hiccup": c.hiccup,
                    }
                    for c in m.llm_calls
                ],
                "tool_calls": [
                    {
                        "turn": t.turn, "tool": t.tool_name,
                        "resolver_ms": round(t.resolver_ms, 4),
                        "routing_ms":  round(t.routing_ms, 2),
                        "e2e_ms":      round(t.e2e_ms, 2),
                        "success":     t.success,
                    }
                    for t in m.tool_calls
                ],
            }
            for m in sorted(all_metrics, key=lambda x: (x.scenario_id, x.agent_id))
        ],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts_tag = run_ts.replace(":", "").replace(" ", "_").replace("-", "")
    fpath  = os.path.join(RESULTS_DIR, f"single_{cfg.key}_{ACTIVE_GPU}_{ts_tag}.json")
    with open(fpath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[results] Written → {fpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# VLLM SERVER  (single instance, NO --data-parallel-size flag)
# ═══════════════════════════════════════════════════════════════════════════════

_NOISE_RE = re.compile(
    r"(Avg prompt throughput:|Avg generation throughput:|Running: \d+ reqs"
    r"|Waiting: \d+ reqs|GPU KV cache usage:|Prefix cache hit rate:"
    r"|HTTP/1\.1\" 200 OK|HTTP/1\.1\" 400)"
)

def _pipe_filter(src: io.RawIOBase, dst) -> None:
    try:
        for raw in src:
            line = raw.decode(errors="replace")
            if not _NOISE_RE.search(line):
                dst.write(raw)
                dst.flush()
    except Exception:
        pass

def _start_vllm(cfg: ModelConfig, port: int) -> subprocess.Popen:
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                cfg.hf_repo,
        "--download-dir",         MODEL_CACHE,
        "--dtype",                cfg.dtype,
        "--tensor-parallel-size", str(cfg.tensor_parallel),
        # NO --data-parallel-size  → single engine instance
        "--max-model-len",        str(cfg.max_model_len),
        "--port",                 str(port),
        "--no-enable-log-requests",
        "--trust-remote-code",
        "--hf-token",             os.environ.get("HF_TOKEN", ""),
        "--attention-backend",    "TRITON_ATTN",
        "--enforce-eager",
        "--gpu-memory-utilization", "0.88",
        # Generous KV-cache budget for 5 concurrent long conversations
        "--max-num-seqs",         "32",
    ]
    if cfg.tool_mode == "native":
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", cfg.tool_call_parser]
    if cfg.quantization:
        cmd += ["--quantization", cfg.quantization]
    cmd += cfg.extra_vllm_args

    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_DISABLE_FLASHINFER"] = "1"
    env["VLLM_USE_V1"] = "0"
    env["VLLM_ATTENTION_BACKEND"]      = "TRITON_ATTN"
    print(f"[vLLM] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    for src in (proc.stdout, proc.stderr):
        threading.Thread(target=_pipe_filter, args=(src, sys.stdout.buffer), daemon=True).start()
    return proc

def _wait_for_vllm(port: int, timeout_s: int = 3600) -> None:
    deadline = time.time() + timeout_s
    url = f"http://localhost:{port}/health"
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            print(f"[vLLM] Server ready on :{port}")
            return
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(3)
    raise RuntimeError(f"vLLM did not become ready within {timeout_s}s")


# ═══════════════════════════════════════════════════════════════════════════════
# BODY
# ═══════════════════════════════════════════════════════════════════════════════

def _body(cfg: ModelConfig, scenario_ids: list[int]) -> None:
    run_ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    proc     = _start_vllm(cfg, VLLM_PORT)
    try:
        _wait_for_vllm(VLLM_PORT)
        base_url    = f"http://localhost:{VLLM_PORT}/v1"
        all_metrics: list[AgentMetrics] = []
        res_lock    = threading.Lock()

        for sid in scenario_ids:
            s = _SCENARIOS[sid]
            print(f"\n{'─'*72}")
            print(f"  [{s['name']}]  {N_AGENTS} agents → 1 model instance")
            print(f"{'─'*72}")

            batch:   list[AgentMetrics]  = []
            threads: list[threading.Thread] = []

            def _run(aid: int, _sid: int = sid, _s: dict = s) -> None:
                m = run_conversation(
                    scenario_name=_s["name"],
                    caller_turns=_s["turns"],
                    caller_type=_s["caller_type"],
                    cfg=cfg,
                    base_url=base_url,
                    agent_id=aid,
                    scenario_id=_sid,
                )
                with res_lock:
                    batch.append(m)

            for aid in range(1, N_AGENTS + 1):
                threads.append(threading.Thread(target=_run, args=(aid,), daemon=True))

            # Stagger by 250 ms to spread initial prefill load
            for t in threads:
                t.start()
                time.sleep(0.25)

            for t in threads:
                t.join()

            all_metrics.extend(batch)

        hiccup_count = _detect_hiccups(all_metrics)
        print_metrics_report(cfg, all_metrics, scenario_ids, hiccup_count)
        _dump_results(cfg, all_metrics, scenario_ids, hiccup_count, run_ts)

    finally:
        proc.terminate()
        proc.wait()


# ═══════════════════════════════════════════════════════════════════════════════
# MODAL FUNCTIONS  (gpu= from truth table, NO shm_size — single engine, no NCCL)
# ═══════════════════════════════════════════════════════════════════════════════

_SEC = [modal.Secret.from_name("huggingface-secret")]
_VOL = {MODEL_CACHE: model_volume, RESULTS_DIR: results_volume}


@app.function(image=image, gpu=_modal_gpu("gpt120b_mxfp4"), volumes=_VOL, timeout=3600, secrets=_SEC)
def run_gpt120b_mxfp4(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gpt120b_mxfp4"], scenario_ids)


@app.function(image=image, gpu=_modal_gpu("gpt120b_bf16"),  volumes=_VOL, timeout=3600, secrets=_SEC)
def run_gpt120b_bf16(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gpt120b_bf16"], scenario_ids)


@app.function(image=image, gpu=_modal_gpu("gemma4_26b"),    volumes=_VOL, timeout=3600, secrets=_SEC)
def run_gemma4_26b(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gemma4_26b"], scenario_ids)


@app.function(image=image, gpu=_modal_gpu("gemma4_31b"),    volumes=_VOL, timeout=3600, secrets=_SEC)
def run_gemma4_31b(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gemma4_31b"], scenario_ids)


@app.function(image=image, gpu=_modal_gpu("qwen3_72b_fp8"), volumes=_VOL, timeout=3600, secrets=_SEC)
def run_qwen3_72b_fp8(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["qwen3_72b_fp8"], scenario_ids)


@app.function(image=image, gpu=_modal_gpu("qwen3_72b_bf16"), volumes=_VOL, timeout=3600, secrets=_SEC)
def run_qwen3_72b_bf16(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["qwen3_72b_bf16"], scenario_ids)


_RUNNERS: dict[str, Any] = {
    "gpt120b_mxfp4":  run_gpt120b_mxfp4,
    "gpt120b_bf16":   run_gpt120b_bf16,
    "gemma4_26b":     run_gemma4_26b,
    "gemma4_31b":     run_gemma4_31b,
    "qwen3_72b_fp8":  run_qwen3_72b_fp8,
    "qwen3_72b_bf16": run_qwen3_72b_bf16,
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(model: str = "", scenario: int = 0) -> None:
    """
    modal run on_prem_oss_unimodel.py
    modal run on_prem_oss_unimodel.py --model gemma4_26b
    modal run on_prem_oss_unimodel.py --model qwen3_72b_fp8 --scenario 2
    modal run on_prem_oss_unimodel.py --scenario 3

    ACTIVE_GPU=B200         modal run on_prem_oss_unimodel.py
    ACTIVE_GPU=RTX_PRO_6000 modal run on_prem_oss_unimodel.py --model gemma4_26b
    N_AGENTS=10             modal run on_prem_oss_unimodel.py --model gemma4_26b
    """
    if model and model not in _RUNNERS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {', '.join(sorted(_RUNNERS))}")

    target_models    = [model] if model else sorted(_RUNNERS)
    all_scenario_ids = sorted(_SCENARIOS)
    target_scenarios = [scenario] if scenario in all_scenario_ids else all_scenario_ids

    W = 72
    print(f"\n{'═'*W}")
    print(f"  NGI Pharma — Single-Model Multi-KV-Cache Demo")
    print(f"  GPU      : {ACTIVE_GPU}  ({_MODAL_GPU_TAG[ACTIVE_GPU]})")
    print(f"  Agents   : {N_AGENTS}  (independent KV-cache contexts, 1 model)")
    print(f"  Models   : {target_models}")
    print(f"  Scenarios: {target_scenarios}")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*W}\n")
    print(f"  GPU allocation (TP only — single server, no DP):")
    print(f"  {'Model':<20} {'TP':>4}  {'GPUs':>6}")
    print(f"  {'─'*34}")
    for key, cfg in MODEL_REGISTRY.items():
        print(f"  {key:<20} {cfg.tensor_parallel:>4}  {cfg.gpu_count:>6}")
    print()

    for mk in target_models:
        cfg = MODEL_REGISTRY[mk]
        print(f"\n{'─'*W}")
        print(f"  {cfg.display_name}")
        print(f"  dtype={cfg.dtype}"
              + (f"  quant={cfg.quantization}" if cfg.quantization else "")
              + f"  tool_mode={cfg.tool_mode}")
        print(f"{'─'*W}\n")
        _RUNNERS[mk].remote(target_scenarios)