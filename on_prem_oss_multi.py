"""
on_prem_oss_multi.py
═══════════════════════════════════════════════════════════════════════════════
NGI Pharma Voice AI — Modal demo runner (5-concurrent-agent edition).

GPU count = TP × DP as derived from the GPU_TRUTH_TABLE below.

Two independent dimensions:
  N_AGENTS  — concurrent client threads (always 5, change freely)
  gpus_per_model — total GPUs in the container, set per model in GPU_TRUTH_TABLE

TP and DP are derived automatically from gpus_per_model:
  TP = max(gpus_per_model // N_AGENTS, 1)   (tensor shards per replica)
  DP = gpus_per_model // TP                 (vLLM engine replicas)

── GPU Selection ─────────────────────────────────────────────────────────────
  Set ACTIVE_GPU at the top of this file to switch between GPU targets.
  All Modal @app.function decorators and ModelConfig.gpu_count fields are
  derived automatically from GPU_TRUTH_TABLE — no other edits needed.

── Metrics collected ─────────────────────────────────────────────────────────
  1. avg tok/s per scenario (generation throughput, averaged over all agents)
  2. avg tok/s across all scenarios
  3. avg tool-call resolver latency  (pure in-process DB time, μs-range)
  4. avg end-to-end tool latency     (resolver + round-trip LLM reasoning time)

── Usage ─────────────────────────────────────────────────────────────────────
    modal run on_prem_oss_multi.py
    modal run on_prem_oss_multi.py --model gemma4_26b
    modal run on_prem_oss_multi.py --model qwen3_72b_fp8 --scenario 2
    modal run on_prem_oss_multi.py --scenario 3

    # Switch GPU target (edit ACTIVE_GPU below, or override via env):
    ACTIVE_GPU=B200 modal run on_prem_oss_multi.py
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
from statistics import mean
from typing import Any

import modal

# ─────────────────────────────────────────────────────────────────────────────
# ① GPU SELECTION & GLOBAL CONFIGS (Imported from gpu_config.py)
# ─────────────────────────────────────────────────────────────────────────────

import os

# ─────────────────────────────────────────────────────────────────────────────
# GPU Hardware target selection: "RTX_PRO_6000" or "B200"
# Can also be overridden at runtime via the ACTIVE_GPU environment variable.
# ─────────────────────────────────────────────────────────────────────────────
ACTIVE_GPU: str = os.environ.get("ACTIVE_GPU", "B200")

# ─────────────────────────────────────────────────────────────────────────────
# Number of concurrent client agents (threads) to launch per scenario.
# This operates independently of the vLLM data_parallel (DP) replicas.
# ─────────────────────────────────────────────────────────────────────────────
N_AGENTS: int = int(os.environ.get("N_AGENTS", "5"))

# ─────────────────────────────────────────────────────────────────────────────
# GPU TRUTH TABLE CONFIGURATION (Model vs GPU)
# Allows global allocation of how many GPUs each model requires.
#
# Hardware Targets:
# - RTX_PRO_6000 (96 GB GDDR7)
# - B200 (141 GB HBM3e)
# ─────────────────────────────────────────────────────────────────────────────
GPU_TRUTH_TABLE: dict[str, dict[str, int]] = {
    #                          RTX_PRO_6000      B200
    "gpt20b_mxfp4":  {"RTX_PRO_6000": 2, "B200": 1},
    "gpt20b_bf16":   {"RTX_PRO_6000": 2, "B200": 1},
    "gpt120b_mxfp4": {"RTX_PRO_6000": 2, "B200": 1},
    "gpt120b_bf16":  {"RTX_PRO_6000": 2, "B200": 1},
    "gemma4_26b":    {"RTX_PRO_6000": 2, "B200": 1},
    "gemma4_31b":    {"RTX_PRO_6000": 2, "B200": 1},
    "qwen3_72b_fp8": {"RTX_PRO_6000": 2, "B200": 1},
    "qwen3_72b_bf16":{"RTX_PRO_6000": 2, "B200": 1},
}

assert ACTIVE_GPU in ("RTX_PRO_6000", "B200"), (
    f"Unknown ACTIVE_GPU={ACTIVE_GPU!r}. Choose 'RTX_PRO_6000' or 'B200'."
)

# Modal GPU tag strings (used in @app.function gpu= parameter)
_MODAL_GPU_TAG: dict[str, str] = {
    "RTX_PRO_6000": "RTX-PRO-6000",
    "B200":         "B200",
}


def _compute_tp_dp(model_key: str) -> tuple[int, int]:
    """
    Dynamically compute (tensor_parallel, data_parallel) based on:
    - Allocated GPU count for the active hardware in GPU_TRUTH_TABLE.
    - Desired agent concurrency (N_AGENTS).
    - Minimum TP constraints for large models to fit in VRAM.
    """
    gpus_allocated = GPU_TRUTH_TABLE[model_key][ACTIVE_GPU]
    
    # Establish min_tp to prevent VRAM OOM on 96GB RTX 6000 cards.
    # Large 120B / 72B BF16 models need at least TP=2 to boot.
    min_tp = 1
    if ACTIVE_GPU == "RTX_PRO_6000":
        if model_key in ("gpt120b_bf16", "qwen3_72b_bf16"):
            min_tp = 2

    # Enforce that TP must be a power of 2 (vLLM requirement: 1, 2, 4, 8, 16)
    # and find the configuration that maximizes GPU usage and targets N_AGENTS.
    best_tp = min_tp
    best_dp = max(1, gpus_allocated // min_tp)
    best_score = (-1, -999999)
    
    for tp in (1, 2, 4, 8, 16):
        if tp < min_tp:
            continue
        dp = gpus_allocated // tp
        if dp == 0:
            continue
            
        used_gpus = tp * dp
        score = (used_gpus, -abs(dp - N_AGENTS))
        if score > best_score:
            best_score = score
            best_tp = tp
            best_dp = dp
            
    return best_tp, best_dp


def _gpu_count(model_key: str) -> int:
    """Return the total number of GPUs allocated from the truth table."""
    return GPU_TRUTH_TABLE[model_key][ACTIVE_GPU]


def _modal_gpu_spec(model_key: str) -> str:
    """Return a Modal gpu= string like 'B200:10' or 'RTX-PRO-6000:5'."""
    tag   = _MODAL_GPU_TAG[ACTIVE_GPU]
    count = _gpu_count(model_key)
    return f"{tag}:{count}"


# ─────────────────────────────────────────────────────────────────────────────
# Modal app + image
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("pharma-agent-oss-demo")

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

model_volume = modal.Volume.from_name("pharma-model-weights", create_if_missing=True)
MODEL_CACHE_PATH = "/model-cache"
VLLM_PORT = 8100


# ─────────────────────────────────────────────────────────────────────────────
# ③ Model registry — TP/DP/gpu_count derived from truth table at import time
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    key: str
    hf_repo: str
    display_name: str
    dtype: str
    quantization: str | None
    tensor_parallel: int        # derived from truth table
    data_parallel: int          # derived from truth table (== N_AGENTS or clamped)
    gpu_count: int              # = tensor_parallel × data_parallel
    max_model_len: int
    tool_mode: str              # "native" | "json"
    tool_call_parser: str = "hermes"
    extra_vllm_args: list[str] = field(default_factory=list)


def _make_cfg(
    key: str,
    hf_repo: str,
    display_name_tpl: str,
    dtype: str,
    quantization: str | None,
    max_model_len: int,
    tool_mode: str,
    tool_call_parser: str = "hermes",
    extra_vllm_args: list[str] | None = None,
) -> ModelConfig:
    tp, dp    = _compute_tp_dp(key)
    gpu_count = _gpu_count(key)      # = total GPUs allocated (TP × DP)
    gpu_label = _MODAL_GPU_TAG[ACTIVE_GPU]
    display   = display_name_tpl.format(
        gpu=gpu_label, tp=tp, dp=dp, n=gpu_count,
        quant=f" {quantization.upper()}" if quantization else " BF16",
    )
    return ModelConfig(
        key=key,
        hf_repo=hf_repo,
        display_name=display,
        dtype=dtype,
        quantization=quantization,
        tensor_parallel=tp,
        data_parallel=dp,
        gpu_count=gpu_count,
        max_model_len=max_model_len,
        tool_mode=tool_mode,
        tool_call_parser=tool_call_parser,
        extra_vllm_args=extra_vllm_args or [],
    )


MODEL_REGISTRY: dict[str, ModelConfig] = {

    "gpt20b_mxfp4": _make_cfg(
        key="gpt20b_mxfp4",
        hf_repo="openai/gpt-oss-20b",
        display_name_tpl="GPT OSS 20B{quant} — {n}× {gpu} (TP={tp} DP={dp})",
        dtype="bfloat16", quantization="mxfp4",
        max_model_len=32768,
        tool_mode="json",
        extra_vllm_args=["--enable-chunked-prefill"],
    ),

    "gpt20b_bf16": _make_cfg(
        key="gpt20b_bf16",
        hf_repo="openai/gpt-oss-20b",
        display_name_tpl="GPT OSS 20B{quant} — {n}× {gpu} (TP={tp} DP={dp})",
        dtype="bfloat16", quantization=None,
        max_model_len=32768,
        tool_mode="json",
        extra_vllm_args=["--enable-chunked-prefill"],
    ),

    "gpt120b_mxfp4": _make_cfg(
        key="gpt120b_mxfp4",
        hf_repo="openai/gpt-oss-120b",
        display_name_tpl="GPT OSS 120B{quant} — {n}× {gpu} (TP={tp} DP={dp})",
        dtype="bfloat16", quantization="mxfp4",
        max_model_len=32768,
        tool_mode="json",
        extra_vllm_args=["--enable-chunked-prefill"],
    ),

    "gpt120b_bf16": _make_cfg(
        key="gpt120b_bf16",
        hf_repo="openai/gpt-oss-120b",
        display_name_tpl="GPT OSS 120B{quant} — {n}× {gpu} (TP={tp} DP={dp})",
        dtype="bfloat16", quantization=None,
        max_model_len=32768,
        tool_mode="json",
        extra_vllm_args=["--enable-chunked-prefill"],
    ),

    "gemma4_26b": _make_cfg(
        key="gemma4_26b",
        hf_repo="google/gemma-3-27b-it",
        display_name_tpl="Gemma 4 26B{quant} — {n}× {gpu} (TP={tp} DP={dp})",
        dtype="bfloat16", quantization=None,
        max_model_len=32768,
        tool_mode="json",
    ),

    "gemma4_31b": _make_cfg(
        key="gemma4_31b",
        hf_repo="google/gemma-3-27b-it",
        display_name_tpl="Gemma 4 31B{quant} — {n}× {gpu} (TP={tp} DP={dp})",
        dtype="bfloat16", quantization=None,
        max_model_len=32768,
        tool_mode="json",
    ),

    "qwen3_72b_fp8": _make_cfg(
        key="qwen3_72b_fp8",
        hf_repo="Qwen/Qwen2.5-72B-Instruct",
        display_name_tpl="Qwen3 72B{quant} — {n}× {gpu} (TP={tp} DP={dp})",
        dtype="bfloat16", quantization="fp8",
        max_model_len=32768,
        tool_mode="native",
    ),

    "qwen3_72b_bf16": _make_cfg(
        key="qwen3_72b_bf16",
        hf_repo="Qwen/Qwen2.5-72B-Instruct",
        display_name_tpl="Qwen3 72B{quant} — {n}× {gpu} (TP={tp} DP={dp})",
        dtype="bfloat16", quantization=None,
        max_model_len=32768,
        tool_mode="native",
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
                "Look up an insurance member by Emirates ID. "
                "Returns policy metadata. Does NOT return member name — "
                "verify via verify_member_name before disclosing protected info."
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
                "Verify the caller's stated name matches the record. "
                "Call AFTER lookup_member. Returns {verified: true/false}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emirates_id":   {"type": "string"},
                    "provided_name": {"type": "string", "description": "Name as spoken by caller."},
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
                    "drug_name":   {"type": "string", "description": "e.g. 'Zocor 40mg'"},
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
                "Get covered alternatives for a drug class with real-time inventory. "
                "Use drug_class from get_claim_status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_class":  {"type": "string", "description": "e.g. 'statin'"},
                    "pharmacy_id": {"type": "string", "description": "e.g. 'DXB-PH-005'"},
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

After you receive the tool result (injected as a user message), continue
the conversation naturally in plain text. If no tool call is needed, respond
normally in plain text.

Available tools:
{_TOOLS_JSON_SCHEMA}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

def execute_tool(name: str, inputs: dict) -> tuple[dict, float]:
    """Execute tool against _DB. Returns (result, resolver_latency_ms)."""
    t0 = time.perf_counter()

    if name == "lookup_member":
        eid = inputs["emirates_id"].strip()
        m = _DB["members"].get(eid)
        result: dict = (
            {"found": False, "error": "No member record."}
            if not m else
            {
                "found": True,
                "emirates_id":      m["emirates_id"],
                "policy_number":    m["policy_number"],
                "insurer":          m["insurer"],
                "plan":             m["plan"],
                "status":           m["status"],
                "copay_pct":        m["copay_pct"],
                "expiry_date":      m["expiry_date"],
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
            match  = (provided == stored) or (provided in stored) or (stored in provided)
            result = {"verified": match}

    elif name == "get_claim_status":
        eid     = inputs["emirates_id"].strip()
        query   = inputs["drug_name"].strip().lower()
        matched = None
        for claim in _DB["claims"]:
            if claim["member_id"] == eid and (
                query in claim["drug"].lower()
                or query in claim["generic"].lower()
                or claim["drug"].lower() in query
            ):
                matched = claim
                break
        result = (
            {"found": False, "message": "No claim found."}
            if not matched else
            {
                "found":            True,
                "claim_id":         matched["claim_id"],
                "drug":             matched["drug"],
                "generic":          matched["generic"],
                "drug_class":       matched["drug_class"],
                "status":           matched["status"],
                "pa_required":      matched["pa_required"],
                "pa_reason":        matched["pa_reason"],
                "rejection_reason": matched["rejection_reason"],
                "submitted":        matched["submitted"],
            }
        )

    elif name == "get_formulary_alternatives":
        dc   = inputs["drug_class"].strip().lower()
        pid  = inputs["pharmacy_id"].strip()
        alts = _DB["formulary_alternatives"].get(dc, [])
        inv  = _DB["inventory"].get(pid, {})
        result = {
            "drug_class":   dc,
            "pharmacy_id":  pid,
            "alternatives": [
                {
                    **a,
                    "inventory_status": inv.get(a["drug"], {}).get("status", "unknown"),
                    "qty_on_hand":      inv.get(a["drug"], {}).get("qty", 0),
                }
                for a in alts
            ],
        }

    elif name == "get_policy_status":
        eid = inputs["emirates_id"].strip()
        m   = _DB["members"].get(eid)
        result = (
            {"found": False}
            if not m else
            {
                "found":          True,
                "policy_number":  m["policy_number"],
                "insurer":        m["insurer"],
                "plan":           m["plan"],
                "status":         m["status"],
                "expiry_date":    m["expiry_date"],
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
2. Call lookup_member first, then verify_member_name. Do not disclose claim or policy
   details until verify_member_name returns {verified: true}.
3. If verification fails: "I'm unable to verify the identity on record." Do NOT reveal stored name.

CLAIM & POLICY RULES
4. Use get_claim_status with the exact drug the caller mentions.
5. When a claim is "under_review" due to PA, explain why PA is required (use pa_reason),
   what must be submitted, and that review takes 24-48 hours after submission.
6. When suggesting alternatives, include inventory status from get_formulary_alternatives.
7. If a policy is "expired", direct the caller to HR or the insurer. Do not process claims.
8. Use get_policy_status as a fast-path check when a claim is rejected.

VOICE BEHAVIOR
- Phone call: keep responses to 2-4 sentences per turn.
- No bullet points or headers.
- Professional, warm, efficient.
- Always use tools to fetch data. Never invent claim status, inventory, copays, or policy details.
"""


def build_system_prompt(tool_mode: str) -> str:
    return _SYSTEM_BASE + ("\n" + _JSON_SHIM_ADDENDUM if tool_mode == "json" else "")


# ═══════════════════════════════════════════════════════════════════════════════
# JSON-SHIM PARSER
# ═══════════════════════════════════════════════════════════════════════════════

_JSON_TOOL_RE = re.compile(
    r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)
_GEMMA_TOOL_RE = re.compile(
    r'```(?:tool_code|python)?\s*\n(\w+)\(([^)]*)\)\s*\n```',
    re.DOTALL,
)


def _parse_gemma_tool_call(text: str) -> dict | None:
    m = _GEMMA_TOOL_RE.search(text)
    if not m:
        return None
    fn_name, args_str = m.group(1), m.group(2).strip()
    args: dict = {}
    for kv in re.finditer(
        r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\S+?)(?:,|$))', args_str
    ):
        args[kv.group(1)] = kv.group(2) or kv.group(3) or kv.group(4)
    return {"name": fn_name, "arguments": args} if fn_name and args else None


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
    return _parse_gemma_tool_call(text)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentMetrics:
    agent_id: int
    scenario_id: int
    tokens_generated: int = 0
    scenario_duration_s: float = 0.0
    resolver_latencies_ms: list[float] = field(default_factory=list)
    e2e_tool_latencies_ms: list[float] = field(default_factory=list)
    tool_log: list[dict] = field(default_factory=list)
    tool_calls_success: int = 0
    tool_calls_failed: int = 0
    unresponded_turns: int = 0

    @property
    def tok_per_s(self) -> float:
        return self.tokens_generated / self.scenario_duration_s if self.scenario_duration_s > 0 else 0.0

    @property
    def avg_resolver_ms(self) -> float:
        return mean(self.resolver_latencies_ms) if self.resolver_latencies_ms else 0.0

    @property
    def avg_e2e_ms(self) -> float:
        return mean(self.e2e_tool_latencies_ms) if self.e2e_tool_latencies_ms else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# VLLM SERVER
# ═══════════════════════════════════════════════════════════════════════════════

_VLLM_NOISE_RE = re.compile(
    r"(APIServer pid=|Avg prompt throughput:|Avg generation throughput:"
    r"|Running: \d+ reqs|Waiting: \d+ reqs|GPU KV cache usage:"
    r"|Prefix cache hit rate:|HTTP/1\.1\" 200 OK)"
)


def _pipe_filter(src: io.RawIOBase, dst: io.RawIOBase) -> None:
    try:
        for raw in src:
            line = raw.decode(errors="replace")
            if not _VLLM_NOISE_RE.search(line):
                dst.write(raw)
                dst.flush()
    except Exception:
        pass


def _start_vllm_server(cfg: ModelConfig, port: int) -> subprocess.Popen:
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  cfg.hf_repo,
        "--download-dir",           MODEL_CACHE_PATH,
        "--dtype",                  cfg.dtype,
        "--tensor-parallel-size",   str(cfg.tensor_parallel),
        "--data-parallel-size",     str(cfg.data_parallel),
        "--max-model-len",          str(cfg.max_model_len),
        "--port",                   str(port),
        "--no-enable-log-requests",
        "--trust-remote-code",
        "--hf-token",               os.environ.get("HF_TOKEN", ""),
        "--attention-backend",      "TRITON_ATTN",
        "--enforce-eager",
        "--gpu-memory-utilization", "0.88",
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
    print(f"[vLLM] Starting: {' '.join(cmd)}")
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
            print(f"[vLLM] Server ready on port {port}")
            return
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(3)
    raise RuntimeError(f"vLLM server did not become ready within {timeout_s}s")


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION RUNNER  (single agent)
# ═══════════════════════════════════════════════════════════════════════════════

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
    lock    = threading.Lock()

    with lock:
        print(f"\n{'═'*72}")
        print(f"  {scenario_name}  [Agent {agent_id}]")
        print(f"  Model  : {cfg.display_name}")
        print(f"  GPU    : {ACTIVE_GPU}  TP={cfg.tensor_parallel}  DP={cfg.data_parallel}  ({cfg.gpu_count} GPUs)")
        print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'═'*72}\n")

    messages: list[dict] = [{"role": "system", "content": build_system_prompt(cfg.tool_mode)}]
    scenario_t0 = time.perf_counter()

    for turn_index, caller_utterance in enumerate(caller_turns, start=1):
        with lock:
            print(f"  [A{agent_id}] TURN {turn_index}: {caller_utterance[:80]}")

        messages.append({"role": "user", "content": caller_utterance})
        responded_this_turn = False

        while True:
            # ── NATIVE tool-calling ──────────────────────────────────────────
            if cfg.tool_mode == "native":
                t_llm0   = time.perf_counter()
                response = client.chat.completions.create(
                    model=cfg.hf_repo,
                    messages=messages,
                    tools=TOOLS_OPENAI,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=512,
                )
                msg   = response.choices[0].message
                usage = response.usage
                if usage:
                    metrics.tokens_generated += (usage.completion_tokens or 0)

                if msg.content and msg.content.strip():
                    responded_this_turn = True
                    with lock:
                        print(f"  [A{agent_id}] AGENT: {msg.content.strip()[:120]}\n")

                if not msg.tool_calls:
                    messages.append({"role": "assistant", "content": msg.content or ""})
                    break

                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id, "type": "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)

                    result, resolver_ms = execute_tool(fn_name, fn_args)
                    e2e_ms  = (time.perf_counter() - t_llm0) * 1_000
                    tool_ok = "error" not in result

                    if tool_ok:
                        metrics.tool_calls_success += 1
                    else:
                        metrics.tool_calls_failed  += 1

                    metrics.resolver_latencies_ms.append(resolver_ms)
                    metrics.e2e_tool_latencies_ms.append(e2e_ms)
                    metrics.tool_log.append({
                        "turn": turn_index, "tool": fn_name,
                        "input": fn_args, "result": result,
                        "resolver_ms": round(resolver_ms, 4),
                        "e2e_ms": round(e2e_ms, 2),
                        "success": tool_ok,
                    })

                    with lock:
                        print(f"  [A{agent_id}] TOOL→ {fn_name}  resolver={resolver_ms:.3f}ms  e2e={e2e_ms:.0f}ms")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            # ── JSON-SHIM tool-calling ───────────────────────────────────────
            else:
                t_llm0   = time.perf_counter()
                response = client.chat.completions.create(
                    model=cfg.hf_repo,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=512,
                )
                usage = response.usage
                if usage:
                    metrics.tokens_generated += (usage.completion_tokens or 0)

                raw_text  = response.choices[0].message.content or ""
                tool_call = parse_json_tool_call(raw_text)

                if tool_call is None:
                    if raw_text.strip():
                        responded_this_turn = True
                        with lock:
                            print(f"  [A{agent_id}] AGENT: {raw_text.strip()[:120]}\n")
                    messages.append({"role": "assistant", "content": raw_text})
                    break

                fn_name = tool_call["name"]
                fn_args = tool_call["arguments"]

                result, resolver_ms = execute_tool(fn_name, fn_args)
                e2e_ms  = (time.perf_counter() - t_llm0) * 1_000
                tool_ok = "error" not in result

                if tool_ok:
                    metrics.tool_calls_success += 1
                else:
                    metrics.tool_calls_failed  += 1

                metrics.resolver_latencies_ms.append(resolver_ms)
                metrics.e2e_tool_latencies_ms.append(e2e_ms)
                metrics.tool_log.append({
                    "turn": turn_index, "tool": fn_name,
                    "input": fn_args, "result": result,
                    "resolver_ms": round(resolver_ms, 4),
                    "e2e_ms": round(e2e_ms, 2),
                    "success": tool_ok,
                })

                with lock:
                    print(f"  [A{agent_id}] TOOL→ {fn_name}  resolver={resolver_ms:.3f}ms  e2e={e2e_ms:.0f}ms")

                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": f"[TOOL RESULT for {fn_name}]: {json.dumps(result, ensure_ascii=False)}",
                })

        if not responded_this_turn:
            metrics.unresponded_turns += 1
            with lock:
                print(f"  [WARNING] [A{agent_id}] TURN {turn_index} had NO agent text response!")

    metrics.scenario_duration_s = time.perf_counter() - scenario_t0
    return metrics


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
# METRICS PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

def print_metrics_report(
    cfg: ModelConfig,
    all_metrics: list[AgentMetrics],
    scenario_ids: list[int],
) -> None:
    sep  = "═" * 72
    sep2 = "─" * 72

    print(f"\n{sep}")
    print(f"  METRICS REPORT — {cfg.display_name}")
    print(f"  GPU target : {ACTIVE_GPU}")
    print(f"  {cfg.gpu_count} GPUs  |  TP={cfg.tensor_parallel}  DP={cfg.data_parallel}  |  {N_AGENTS} concurrent agents")
    print(f"{sep}\n")

    print(f"  {'Scenario':<10} {'Agents':>6} {'Avg tok/s':>10} {'Avg res ms':>11} {'Avg e2e ms':>11} {'OK':>5} {'Fail':>5}")
    print(f"  {sep2}")

    all_toks: list[float] = []
    all_res:  list[float] = []
    all_e2e:  list[float] = []
    total_ok   = 0
    total_fail = 0

    for sid in scenario_ids:
        group = [m for m in all_metrics if m.scenario_id == sid]
        if not group:
            continue
        tps  = [m.tok_per_s          for m in group]
        res  = [r for m in group for r in m.resolver_latencies_ms]
        e2e  = [e for m in group for e in m.e2e_tool_latencies_ms]
        ok   = sum(m.tool_calls_success for m in group)
        fail = sum(m.tool_calls_failed  for m in group)

        avg_tps = mean(tps) if tps else 0.0
        avg_res = mean(res) if res else 0.0
        avg_e2e = mean(e2e) if e2e else 0.0

        all_toks.extend(tps);  all_res.extend(res);  all_e2e.extend(e2e)
        total_ok += ok;        total_fail += fail

        print(f"  Scenario {sid:<2}  {len(group):>6}  {avg_tps:>9.1f}  {avg_res:>10.3f}  {avg_e2e:>10.1f}  {ok:>4}  {fail:>4}")

    print(f"  {sep2}")

    total_calls = sum(len(m.tool_log) for m in all_metrics)
    print(f"\n  {'Metric':<46} {'Value':>12}")
    print(f"  {sep2}")
    print(f"  {'1. Avg tok/s per scenario (mean of agents)':<46} {mean(all_toks):>11.1f}")
    print(f"  {'2. Avg tok/s all scenarios combined':<46} {mean(all_toks):>11.1f}")
    print(f"  {'3. Avg tool resolver latency (DB only)':<46} {mean(all_res) if all_res else 0.0:>10.3f} ms")
    print(f"  {'4. Avg tool e2e latency (resolver+LLM)':<46} {mean(all_e2e) if all_e2e else 0.0:>10.1f} ms")
    print(f"  {'5. Total tool calls':<46} {total_calls:>12}")
    print(f"  {'   ↳ Successful (no error key in result)':<46} {total_ok:>12}")
    print(f"  {'   ↳ Failed (error key present)':<46} {total_fail:>12}")
    pct = (total_ok / total_calls * 100) if total_calls else 0.0
    print(f"  {'   ↳ Success rate':<46} {pct:>10.1f} %")
    total_unresponded = sum(m.unresponded_turns for m in all_metrics)
    print(f"  {'6. Total unresponded turns (agent failed to text)':<46} {total_unresponded:>12}")
    print(f"\n{sep}\n")

    print(f"  PER-AGENT TOOL CALL DETAIL")
    print(f"  {'Agent':<7} {'Scen':>5} {'Turns tok':>10} {'tok/s':>7} {'Tools':>6} {'OK':>4} {'Fail':>4} {'NoResp':>6} {'Res ms':>9} {'E2E ms':>9}")
    print(f"  {sep2}")
    for m in sorted(all_metrics, key=lambda x: (x.scenario_id, x.agent_id)):
        res_avg = mean(m.resolver_latencies_ms) if m.resolver_latencies_ms else 0.0
        e2e_avg = mean(m.e2e_tool_latencies_ms) if m.e2e_tool_latencies_ms else 0.0
        print(
            f"  A{m.agent_id:<6} {m.scenario_id:>5} {m.tokens_generated:>10} "
            f"{m.tok_per_s:>7.1f} {len(m.tool_log):>6} {m.tool_calls_success:>4} "
            f"{m.tool_calls_failed:>4} {m.unresponded_turns:>6} {res_avg:>9.3f} {e2e_avg:>9.1f}"
        )
    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS SERIALISER
# ═══════════════════════════════════════════════════════════════════════════════

def _build_results_payload(
    cfg: ModelConfig,
    all_metrics: list[AgentMetrics],
    scenario_ids: list[int],
    run_ts: str,
) -> dict:
    all_toks   = [m.tok_per_s for m in all_metrics]
    all_res    = [r for m in all_metrics for r in m.resolver_latencies_ms]
    all_e2e    = [e for m in all_metrics for e in m.e2e_tool_latencies_ms]
    total_ok   = sum(m.tool_calls_success for m in all_metrics)
    total_fail = sum(m.tool_calls_failed  for m in all_metrics)
    total_calls = total_ok + total_fail

    per_scenario = []
    for sid in scenario_ids:
        group = [m for m in all_metrics if m.scenario_id == sid]
        if not group:
            continue
        tps  = [m.tok_per_s for m in group]
        res  = [r for m in group for r in m.resolver_latencies_ms]
        e2e  = [e for m in group for e in m.e2e_tool_latencies_ms]
        ok   = sum(m.tool_calls_success for m in group)
        fail = sum(m.tool_calls_failed  for m in group)
        unresp = sum(m.unresponded_turns for m in group)
        per_scenario.append({
            "scenario_id":            sid,
            "scenario_name":          _SCENARIOS[sid]["name"],
            "agents":                 len(group),
            "avg_tok_per_s":          round(mean(tps), 3) if tps else 0.0,
            "avg_resolver_ms":        round(mean(res), 4) if res else 0.0,
            "avg_e2e_ms":             round(mean(e2e), 2) if e2e else 0.0,
            "tool_calls_success":     ok,
            "tool_calls_failed":      fail,
            "tool_success_rate_pct":  round(ok / (ok + fail) * 100, 1) if (ok + fail) else 0.0,
            "unresponded_turns":      unresp,
        })

    per_agent = [
        {
            "agent_id":            m.agent_id,
            "scenario_id":         m.scenario_id,
            "tokens_generated":    m.tokens_generated,
            "tok_per_s":           round(m.tok_per_s, 3),
            "scenario_duration_s": round(m.scenario_duration_s, 3),
            "tool_calls_success":  m.tool_calls_success,
            "tool_calls_failed":   m.tool_calls_failed,
            "unresponded_turns":   m.unresponded_turns,
            "avg_resolver_ms":     round(m.avg_resolver_ms, 4),
            "avg_e2e_ms":          round(m.avg_e2e_ms, 2),
            "tool_log":            m.tool_log,
        }
        for m in sorted(all_metrics, key=lambda x: (x.scenario_id, x.agent_id))
    ]

    return {
        "run_timestamp":      run_ts,
        "active_gpu":         ACTIVE_GPU,
        "model_key":          cfg.key,
        "model_display_name": cfg.display_name,
        "hf_repo":            cfg.hf_repo,
        "gpu_count":          cfg.gpu_count,
        "tensor_parallel":    cfg.tensor_parallel,
        "data_parallel":      cfg.data_parallel,
        "dtype":              cfg.dtype,
        "quantization":       cfg.quantization,
        "tool_mode":          cfg.tool_mode,
        "max_model_len":      cfg.max_model_len,
        "n_agents":           N_AGENTS,
        "scenario_ids":       scenario_ids,
        "summary": {
            "avg_tok_per_s":         round(mean(all_toks), 3) if all_toks else 0.0,
            "avg_resolver_ms":       round(mean(all_res),  4) if all_res  else 0.0,
            "avg_e2e_ms":            round(mean(all_e2e),  2) if all_e2e  else 0.0,
            "total_tool_calls":      total_calls,
            "tool_calls_success":    total_ok,
            "tool_calls_failed":     total_fail,
            "tool_success_rate_pct": round(total_ok / total_calls * 100, 1) if total_calls else 0.0,
            "total_unresponded_turns": sum(m.unresponded_turns for m in all_metrics),
        },
        "per_scenario": per_scenario,
        "per_agent":    per_agent,
    }


def _dump_results(payload: dict, cfg: ModelConfig, run_ts: str, out_dir: str = "/results") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts_tag = run_ts.replace(":", "").replace(" ", "_").replace("-", "")
    fpath  = os.path.join(out_dir, f"results_{cfg.key}_{ACTIVE_GPU}_{ts_tag}.json")
    with open(fpath, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[results] Written → {fpath}")
    return fpath


# ═══════════════════════════════════════════════════════════════════════════════
# BODY  (runs inside each Modal container)
# ═══════════════════════════════════════════════════════════════════════════════

def _body(cfg: ModelConfig, scenario_ids: list[int]) -> None:
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Reconcile vLLM DP with physically visible GPUs ───────────────────────
    # vLLM crashes with "DP adjusted local rank N out of bounds" if the number
    # of DP replicas it tries to spawn exceeds the GPUs in the container.
    # We clamp the SERVER-SIDE DP only — N_AGENTS client threads are NEVER
    # reduced; multiple threads will share a replica, which vLLM handles fine.
    try:
        import torch
        visible_gpus = torch.cuda.device_count()
    except Exception:
        visible_gpus = cfg.gpu_count  # torch unavailable (e.g. local dry-run)

    max_dp = max(visible_gpus // max(cfg.tensor_parallel, 1), 1)
    if cfg.data_parallel > max_dp:
        print(
            f"[_body] WARNING: config DP={cfg.data_parallel} but only "
            f"{visible_gpus} GPU(s) visible (TP={cfg.tensor_parallel}). "
            f"Clamping vLLM data_parallel to {max_dp}. "
            f"N_AGENTS={N_AGENTS} client threads still run — they will share replicas."
        )
        cfg.data_parallel = max_dp
        cfg.gpu_count     = max_dp * cfg.tensor_parallel

    # N_AGENTS is always the number of client threads, regardless of DP.
    print(
        f"[_body] vLLM: TP={cfg.tensor_parallel}  DP={cfg.data_parallel}  "
        f"({cfg.gpu_count} GPUs)  |  Client agents: {N_AGENTS}"
    )

    proc = _start_vllm_server(cfg, VLLM_PORT)
    try:
        _wait_for_vllm(VLLM_PORT)
        base_url    = f"http://localhost:{VLLM_PORT}/v1"
        all_metrics: list[AgentMetrics] = []
        res_lock    = threading.Lock()

        for sid in scenario_ids:
            s = _SCENARIOS[sid]
            print(f"\n{'─'*72}")
            print(f"  Launching {N_AGENTS} concurrent agents for {s['name']}")
            print(f"{'─'*72}\n")

            threads: list[threading.Thread] = []
            batch:   list[AgentMetrics]     = []

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

            for t in threads:
                t.start()
                time.sleep(0.5)   # stagger to avoid thundering herd

            for t in threads:
                t.join()

            all_metrics.extend(batch)

        print_metrics_report(cfg, all_metrics, scenario_ids)
        payload = _build_results_payload(cfg, all_metrics, scenario_ids, run_ts)
        _dump_results(payload, cfg, run_ts)

    finally:
        proc.terminate()
        proc.wait()


# ═══════════════════════════════════════════════════════════════════════════════
# ④ MODAL FUNCTIONS — gpu= string derived from GPU_TRUTH_TABLE at import time
#    Adding a new model: add a row to GPU_TRUTH_TABLE + MODEL_REGISTRY above,
#    then add one @app.function + def run_<key> block below. Nothing else changes.
# ═══════════════════════════════════════════════════════════════════════════════

_SECRETS = [modal.Secret.from_name("huggingface-secret")]
_VOL     = {MODEL_CACHE_PATH: model_volume}


@app.function(image=image, gpu=_modal_gpu_spec("gpt20b_mxfp4"), volumes=_VOL, timeout=3600, secrets=_SECRETS)
def run_gpt20b_mxfp4(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gpt20b_mxfp4"], scenario_ids)


@app.function(image=image, gpu=_modal_gpu_spec("gpt20b_bf16"), volumes=_VOL, timeout=3600, secrets=_SECRETS)
def run_gpt20b_bf16(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gpt20b_bf16"], scenario_ids)


@app.function(image=image, gpu=_modal_gpu_spec("gpt120b_mxfp4"), volumes=_VOL, timeout=3600, secrets=_SECRETS)
def run_gpt120b_mxfp4(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gpt120b_mxfp4"], scenario_ids)


@app.function(image=image, gpu=_modal_gpu_spec("gpt120b_bf16"), volumes=_VOL, timeout=3600, secrets=_SECRETS)
def run_gpt120b_bf16(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gpt120b_bf16"], scenario_ids)


@app.function(image=image, gpu=_modal_gpu_spec("gemma4_26b"), volumes=_VOL, timeout=3600, secrets=_SECRETS)
def run_gemma4_26b(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gemma4_26b"], scenario_ids)


@app.function(image=image, gpu=_modal_gpu_spec("gemma4_31b"), volumes=_VOL, timeout=3600, secrets=_SECRETS)
def run_gemma4_31b(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gemma4_31b"], scenario_ids)


@app.function(image=image, gpu=_modal_gpu_spec("qwen3_72b_fp8"), volumes=_VOL, timeout=3600, secrets=_SECRETS)
def run_qwen3_72b_fp8(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["qwen3_72b_fp8"], scenario_ids)


@app.function(image=image, gpu=_modal_gpu_spec("qwen3_72b_bf16"), volumes=_VOL, timeout=3600, secrets=_SECRETS)
def run_qwen3_72b_bf16(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["qwen3_72b_bf16"], scenario_ids)


_RUNNERS: dict[str, Any] = {
    "gpt20b_mxfp4":  run_gpt20b_mxfp4,
    "gpt20b_bf16":   run_gpt20b_bf16,
    "gpt120b_mxfp4": run_gpt120b_mxfp4,
    "gpt120b_bf16":  run_gpt120b_bf16,
    "gemma4_26b":    run_gemma4_26b,
    "gemma4_31b":    run_gemma4_31b,
    "qwen3_72b_fp8": run_qwen3_72b_fp8,
    "qwen3_72b_bf16":run_qwen3_72b_bf16,
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(model: str = "", scenario: int = 0) -> None:
    """
    Run 5-concurrent-agent scenarios on one or all models.

        modal run on_prem_oss_multi.py
        modal run on_prem_oss_multi.py --model gemma4_26b
        modal run on_prem_oss_multi.py --model qwen3_72b_fp8 --scenario 2
        modal run on_prem_oss_multi.py --scenario 3

    Switch GPU target (all gpu= specs and TP/DP configs update automatically):

        ACTIVE_GPU=B200 modal run on_prem_oss_multi.py
        ACTIVE_GPU=RTX_PRO_6000 modal run on_prem_oss_multi.py --model qwen3_72b_fp8
    """
    if model and model not in _RUNNERS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {', '.join(sorted(_RUNNERS))}")

    target_models    = [model] if model else sorted(_RUNNERS)
    all_scenario_ids = sorted(_SCENARIOS)
    target_scenarios = [scenario] if scenario in all_scenario_ids else all_scenario_ids

    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  NGI Pharma OSS Voice AI — {N_AGENTS}-Agent Concurrent Demo")
    print(f"  GPU target  : {ACTIVE_GPU}  ({_MODAL_GPU_TAG[ACTIVE_GPU]})")
    print(f"  Models      : {target_models}")
    print(f"  Scenarios   : {target_scenarios}")
    print(f"  Agents      : {N_AGENTS} concurrent per scenario")
    print(f"  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{sep}")

    # Print truth table for this GPU target
    print(f"\n  GPU Allocation Table ({ACTIVE_GPU})")
    print(f"  {'Model':<20} {'TP':>4} {'DP':>4} {'GPUs':>6}")
    print(f"  {'─'*40}")
    for key, cfg in MODEL_REGISTRY.items():
        print(f"  {key:<20} {cfg.tensor_parallel:>4} {cfg.data_parallel:>4} {cfg.gpu_count:>6}")
    print()

    for model_key in target_models:
        cfg = MODEL_REGISTRY[model_key]
        print(f"\n{'─'*72}")
        print(f"  {cfg.display_name}")
        print(f"  GPUs : {cfg.gpu_count}× {_MODAL_GPU_TAG[ACTIVE_GPU]}")
        print(f"  TP={cfg.tensor_parallel}  DP={cfg.data_parallel}  dtype={cfg.dtype}"
              + (f"  quant={cfg.quantization}" if cfg.quantization else ""))
        print(f"  tool_mode={cfg.tool_mode}")
        print(f"{'─'*72}\n")
        _RUNNERS[model_key].remote(target_scenarios)