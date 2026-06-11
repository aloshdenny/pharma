"""
on_prem.py
═══════════════════════════════════════════════════════════════════════════════
NGI Pharma Voice AI — Full voice pipeline benchmark.

Architecture
────────────
  Shared (1 instance):
    • GPT OSS LLM  via vLLM OpenAI-compatible server (streaming)

  Shared model pools (2 instances each, semaphore-guarded):
    • Whisper Large V3   — STT  (faster-whisper, CUDA, float16)
    • Kokoro-82M         — TTS  (kokoro>=0.9.4, CUDA, float16)

  Why a pool of 2 instead of 5 per-agent instances:
    - 5 × Whisper Large V3 = ~15 GB VRAM; pool of 2 = ~6 GB
    - Freed VRAM goes to vLLM KV-cache (raises --gpu-memory-utilization)
    - At 5 concurrent agents, average queue wait is ~30-60 ms — negligible
      compared to the savings from pipelining

Pipeline per turn (perceived silence gap):

  BEFORE (sequential):
    STT → wait full LLM decode → TTS first chunk
    ≈ 350 ms  +  2670 ms  +  400 ms  =  ~3420 ms  (best case)
    ≈ 350 ms  +  4500 ms  +  400 ms  =  ~5250 ms  (slow model)

  AFTER (pipelined + optimized):
    STT (float16, beam=1) → LLM stream → [first sentence ready] → TTS starts
    ≈ 150 ms  +  2670 ms  +  0 ms*  =  ~2820 ms  (best case)
    ≈ 150 ms  +  4500 ms  +  0 ms*  =  ~4650 ms  (slow model)
    * TTS first-chunk latency is hidden inside the LLM decode window

  Target: avg per-agent e2e (stt + llm + tts_first) < 6 000 ms ✓

Latency breakdown measured:
  stt_ms         — Whisper transcription (WAV → text)
  ttft_ms        — LLM time-to-first-token
  generation_ms  — LLM decode time
  llm_e2e_ms     — full LLM round-trip (all tool loops)
  tts_first_ms   — Kokoro time-to-first-audio-chunk
  tts_full_ms    — Kokoro full synthesis time
  pipeline_ms    — stt_ms + llm_e2e_ms + tts_first_ms  (perceived silence)
  resolver_ms    — pure in-process DB lookup (sub-ms)
  routing_ms     — llm_e2e_ms − resolver_ms  (LLM + tool overhead, no DB)

Warmup:
  All models (Whisper, Kokoro, vLLM) run a single dummy pass before
  scenarios start, so cold-start TTFT is never included in measurements.

GPU allocation (RTX PRO 6000, 96 GB GDDR7):
  LLM vLLM  : ~75 GB  (gpu-memory-utilization=0.78)
  Whisper×2 : ~6  GB
  Kokoro×2  : ~1  GB
  Overhead  : ~5  GB  headroom / fragmentation
  Total     : ~87 GB  (safe on 96 GB)

Usage
─────
    modal run on_prem.py
    modal run on_prem.py --model gemma4_26b
    modal run on_prem.py --model qwen3_72b_fp8 --scenario 2
    ACTIVE_GPU=B200 modal run on_prem.py
    N_AGENTS=10     modal run on_prem.py --model gemma4_26b
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Generator

import modal

# ─────────────────────────────────────────────────────────────────────────────
# Runtime configuration
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_GPU:         str   = os.environ.get("ACTIVE_GPU", "RTX_PRO_6000")
N_AGENTS:           int   = int(os.environ.get("N_AGENTS", "2"))
VLLM_PORT:          int   = 8100
AUDIO_DIR:          str   = "/tmp/pharma_audio"
KOKORO_VOICE:       str   = "af_bella"
KOKORO_SAMPLE_RATE: int   = 24_000

# Model pool sizes — default to N_AGENTS so each agent has dedicated instances
WHISPER_POOL_SIZE:  int   = int(os.environ.get("WHISPER_POOL_SIZE", str(N_AGENTS)))
KOKORO_POOL_SIZE:   int   = int(os.environ.get("KOKORO_POOL_SIZE",  str(N_AGENTS)))

# Sentence streaming: minimum characters before we hand a chunk to TTS.
# Lower = TTS starts sooner (more overlap savings) but more short synthesis calls.
# 30 chars ≈ half a short sentence — safe starting point for streaming.
TTS_STREAM_MIN_CHARS: int = int(os.environ.get("TTS_STREAM_MIN_CHARS", "30"))

# LLM response token budget.  Voice turns are 2-4 sentences ≈ 80-150 tokens;
# 200 gives headroom without paying for 512-token decode windows.
LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "200"))

# Maximum tool-call loops per turn before the agent is forced to respond in plain text.
# Guards against infinite tool loops (e.g. model confused by garbled STT).
MAX_TOOL_LOOPS: int = int(os.environ.get("MAX_TOOL_LOOPS", "8"))

assert ACTIVE_GPU in ("RTX_PRO_6000", "B200"), (
    f"Unknown ACTIVE_GPU={ACTIVE_GPU!r}. Choose 'RTX_PRO_6000' or 'B200'."
)

_MODAL_GPU_TAG: dict[str, str] = {
    "RTX_PRO_6000": "RTX-PRO-6000",
    "B200":         "B200",
}

# ─────────────────────────────────────────────────────────────────────────────
# Modal app + image
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("pharma-voice-pipeline-demo")

_CUDA_IMAGE = (
    "nvidia/cuda:12.8.1-devel-ubuntu22.04"
    if ACTIVE_GPU == "B200"
    else "nvidia/cuda:12.4.1-devel-ubuntu22.04"
)

image = (
    modal.Image.from_registry(_CUDA_IMAGE, add_python="3.11")
    .apt_install("espeak-ng", "ffmpeg", "libsndfile1")
    .pip_install(
        "openai>=1.30.0",
        "faster-whisper>=1.0.0",
        "kokoro>=0.9.4",
        "soundfile>=0.12.1",
        "scipy>=1.11.0",
        "vllm>=0.6.0",
        "huggingface-hub>=0.23.0",
        "numpy>=1.24.0",
    )
    .env({"VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
)

model_volume   = modal.Volume.from_name("pharma-model-weights",  create_if_missing=True)
results_volume = modal.Volume.from_name("pharma-results",        create_if_missing=True)
MODEL_CACHE    = "/model-cache"
RESULTS_DIR    = "/results"

# ─────────────────────────────────────────────────────────────────────────────
# GPU truth table
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# GPU truth table  (TP = tensor-parallel, DP = data-parallel / pipeline-parallel)
# Total GPUs allocated per model = TP × DP.
#
# DP > 1 runs multiple model replicas, each handling a subset of sequences.
# This halves KV-cache pressure per replica — the primary bottleneck at 5+
# concurrent agents on large models.  Set DP=2 if you have 2× the GPU budget.
# ─────────────────────────────────────────────────────────────────────────────

_TP_TABLE: dict[str, dict[str, int]] = {
    #                          RTX_PRO_6000  B200
    "gpt120b_mxfp4":           {"RTX_PRO_6000": 1, "B200": 1},
    "gpt120b_bf16":            {"RTX_PRO_6000": 1, "B200": 1},
    "gpt20b_bf16":            {"RTX_PRO_6000": 1, "B200": 1},
    "gpt20b_mxfp4":            {"RTX_PRO_6000": 1, "B200": 1},
    "gemma4_26b":              {"RTX_PRO_6000": 1, "B200": 1},
    "gemma4_31b":              {"RTX_PRO_6000": 1, "B200": 1},
    "qwen3_72b_fp8":           {"RTX_PRO_6000": 1, "B200": 1},
    "qwen3_72b_bf16":          {"RTX_PRO_6000": 1, "B200": 1},
}

_DP_TABLE: dict[str, dict[str, int]] = {
    # We now run a single vLLM model shared by all agents on GPU 2,
    # so DP is always 1.
    "gpt120b_mxfp4":           {"RTX_PRO_6000": 1, "B200": 1},
    "gpt120b_bf16":            {"RTX_PRO_6000": 1, "B200": 1},
    "gpt20b_bf16":            {"RTX_PRO_6000": 1, "B200": 1},
    "gpt20b_mxfp4":            {"RTX_PRO_6000": 1, "B200": 1},
    "gemma4_26b":              {"RTX_PRO_6000": 1, "B200": 1},
    "gemma4_31b":              {"RTX_PRO_6000": 1, "B200": 1},
    "qwen3_72b_fp8":           {"RTX_PRO_6000": 1, "B200": 1},
    "qwen3_72b_bf16":          {"RTX_PRO_6000": 1, "B200": 1},
}

def _tp(key: str) -> int:
    # Always allocate exactly 1 GPU for tensor parallel to fit within our strict 2-GPU budget
    # (1 GPU dedicated for audio models, 1 GPU dedicated for vLLM).
    return 1

def _dp(key: str) -> int:
    # Always allocate exactly 1 GPU for data parallel to fit within our strict 2-GPU budget.
    return 1

def _modal_gpu(key: str) -> str:
    # Always allocate exactly 2 GPUs on Modal (GPU 0 for audio STT/TTS, GPU 1 for vLLM).
    return f"{_MODAL_GPU_TAG[ACTIVE_GPU]}:2"

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
    tensor_parallel: int
    data_parallel: int
    max_model_len: int
    tool_mode: str
    tool_call_parser: str = "hermes"
    extra_vllm_args: list[str] = field(default_factory=list)

    @property
    def gpu_count(self) -> int:
        return (self.tensor_parallel * self.data_parallel) + 1


def _cfg(key, hf_repo, dtype, quant, max_len, tool_mode,
         tcp="hermes", extra=None) -> ModelConfig:
    tp  = _tp(key)
    dp  = _dp(key)
    tag = _MODAL_GPU_TAG[ACTIVE_GPU]
    q   = f" {quant.upper()}" if quant else " BF16"
    return ModelConfig(
        key=key, hf_repo=hf_repo,
        display_name=f"{key}{q} — {tp}× {tag} (TP={tp} DP={dp})",
        dtype=dtype, quantization=quant,
        tensor_parallel=tp, data_parallel=dp,
        max_model_len=max_len, tool_mode=tool_mode,
        tool_call_parser=tcp, extra_vllm_args=extra or [],
    )


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt120b_mxfp4": _cfg("gpt120b_mxfp4", "openai/gpt-oss-120b",
                           "bfloat16", "mxfp4", 4096, "json",
                           extra=["--enable-chunked-prefill"]),
    "gpt120b_bf16":  _cfg("gpt120b_bf16",  "openai/gpt-oss-120b",
                           "bfloat16", None,   4096, "json",
                           extra=["--enable-chunked-prefill"]),
    "gpt20b_bf16":   _cfg("gpt20b_bf16",   "openai/gpt-oss-20b",
                            "bfloat16", None,   4096, "json",
                            extra=["--enable-chunked-prefill"]),
    "gpt20b_mxfp4":  _cfg("gpt20b_mxfp4",  "openai/gpt-oss-20b",
                            "bfloat16", "mxfp4", 4096, "json",
                            extra=["--enable-chunked-prefill"]),
    "gemma4_26b":    _cfg("gemma4_26b",    "google/gemma-3-27b-it",
                           "bfloat16", None,   4096, "json",
                           extra=["--enable-chunked-prefill"]),
    "gemma4_31b":    _cfg("gemma4_31b",    "google/gemma-3-27b-it",
                           "bfloat16", None,   4096, "json",
                           extra=["--enable-chunked-prefill"]),
    "qwen3_72b_fp8": _cfg("qwen3_72b_fp8", "Qwen/Qwen2.5-72B-Instruct",
                           "bfloat16", "fp8",  4096, "native",
                           extra=["--enable-chunked-prefill"]),
    "qwen3_72b_bf16":_cfg("qwen3_72b_bf16","Qwen/Qwen2.5-72B-Instruct",
                           "bfloat16", None,   4096, "native",
                           extra=["--enable-chunked-prefill"]),
}


# ═══════════════════════════════════════════════════════════════════════════════
# FAKE IN-MEMORY DATABASE  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

_DB: dict[str, Any] = {
    "members": {
        "784-1996-7169603-3": {
            "emirates_id": "784-1996-7169603-3", "name": "Omar Ali",
            "dob": "1996-05-15", "policy_number": "ADNIC-ENH-001",
            "insurer": "ADNIC", "plan": "ADNIC Enhanced", "plan_tier": "Mid",
            "status": "active", "copay_pct": 20, "annual_limit_aed": 300_000,
            "remaining_benefit_aed": 241_500, "network_pharmacy": "DXB-PH-005",
            "expiry_date": None, "policy_start_date": "2024-06-01",
            "policy_end_date": "2025-05-31",
        },
        "784-2004-2137407-6": {
            "emirates_id": "784-2004-2137407-6", "name": "Ahmed Khan",
            "dob": "1988-03-16", "policy_number": "NAS-ENH-042",
            "insurer": "NAS", "plan": "NAS Enhanced", "plan_tier": "Mid",
            "status": "active", "copay_pct": 10, "annual_limit_aed": 300_000,
            "remaining_benefit_aed": 287_400, "network_pharmacy": None,
            "expiry_date": None, "policy_start_date": "2024-04-15",
            "policy_end_date": "2025-04-14",
        },
        "784-1983-4821093-1": {
            "emirates_id": "784-1983-4821093-1", "name": "Hana Patel",
            "dob": "1993-04-21", "policy_number": "ADNIC-ENH-077",
            "insurer": "ADNIC", "plan": "ADNIC Enhanced", "plan_tier": "Mid",
            "status": "active", "copay_pct": 10, "annual_limit_aed": 300_000,
            "remaining_benefit_aed": 280_856, "network_pharmacy": "DXB-PH-022",
            "expiry_date": None, "policy_start_date": "2024-11-06",
            "policy_end_date": "2025-11-05",
        },
        "784-1978-6329401-7": {
            "emirates_id": "784-1978-6329401-7", "name": "Ravi Reyes",
            "dob": "1978-02-08", "policy_number": "DAMAN-GLD-199",
            "insurer": "Daman", "plan": "Daman Gold", "plan_tier": "High",
            "status": "active", "copay_pct": 5, "annual_limit_aed": 400_000,
            "remaining_benefit_aed": 142_163, "network_pharmacy": "DXB-PH-005",
            "expiry_date": None, "policy_start_date": "2024-11-20",
            "policy_end_date": "2025-11-19",
        },
        "784-1985-7741823-5": {
            "emirates_id": "784-1985-7741823-5", "name": "Deepa Ali",
            "dob": "1985-01-20", "policy_number": "AXA-BSC-304",
            "insurer": "AXA Gulf", "plan": "AXA Basic", "plan_tier": "Basic",
            "status": "active", "copay_pct": 20, "annual_limit_aed": 150_000,
            "remaining_benefit_aed": 45_921, "network_pharmacy": "DXB-PH-029",
            "expiry_date": None, "policy_start_date": "2024-12-05",
            "policy_end_date": "2025-12-04",
        },
        "784-1983-5524190-4": {
            "emirates_id": "784-1983-5524190-4", "name": "Nadia Ibrahim",
            "dob": "2002-02-26", "policy_number": "DAMAN-THQ-088",
            "insurer": "Daman", "plan": "Daman Thiqa", "plan_tier": "Premium",
            "status": "active", "copay_pct": 0, "annual_limit_aed": 500_000,
            "remaining_benefit_aed": 291_328, "network_pharmacy": "DXB-PH-029",
            "expiry_date": None, "policy_start_date": "2024-09-27",
            "policy_end_date": "2025-09-26",
        },
        "784-1974-3341057-2": {
            "emirates_id": "784-1974-3341057-2", "name": "Fatima Al Mansoori",
            "dob": "1982-05-05", "policy_number": "CIGNA-ME-117",
            "insurer": "Cigna ME", "plan": "Cigna ME Standard", "plan_tier": "Basic",
            "status": "expired", "copay_pct": 0, "annual_limit_aed": 150_000,
            "remaining_benefit_aed": 0, "network_pharmacy": None,
            "expiry_date": "2024-12-16", "policy_start_date": "2023-12-17",
            "policy_end_date": "2024-12-16",
        },
    },
    "claims": [
        {"claim_id": "CLM-2025-0441", "member_id": "784-1996-7169603-3",
         "drug": "Zocor 40mg", "generic": "Simvastatin 40mg", "drug_class": "statin",
         "status": "under_review", "pa_required": True,
         "pa_reason": "Step therapy applies — documentation of prior failed therapy "
                      "with Simvastatin or Lovastatin required before this brand is approved.",
         "rejection_reason": None, "submitted": "2025-05-20"},
        {"claim_id": "CLM-2025-0512", "member_id": "784-2004-2137407-6",
         "drug": "Januvia 100mg", "generic": "Sitagliptin 100mg",
         "drug_class": "DPP-4 inhibitor", "status": "under_review", "pa_required": True,
         "pa_reason": "Prior Authorization required per NAS formulary Tier 3 policy. "
                      "Physician must submit PA form with clinical notes via E-Claim portal.",
         "rejection_reason": None, "submitted": "2025-05-22"},
        {"claim_id": "CLM-2025-0490", "member_id": "784-2004-2137407-6",
         "drug": "Metformin 500mg", "generic": "Metformin 500mg", "drug_class": "biguanide",
         "status": "approved", "pa_required": False,
         "pa_reason": None, "rejection_reason": None, "submitted": "2025-05-10"},
        {"claim_id": "CLM-2025-0530", "member_id": "784-1974-3341057-2",
         "drug": "Plavix", "generic": "Clopidogrel 75mg", "drug_class": "antiplatelet",
         "status": "rejected", "pa_required": False, "pa_reason": None,
         "rejection_reason": "Policy expired on 2024-12-16; no active coverage.",
         "submitted": "2025-05-23"},
        {"claim_id": "CLM-2025-0601", "member_id": "784-1983-4821093-1",
         "drug": "Lantus", "generic": "Insulin Glargine", "drug_class": "insulin",
         "status": "under_review", "pa_required": True,
         "pa_reason": "Insulin Glargine (Lantus) requires PA under ADNIC Enhanced plan. "
                      "Physician must submit clinical justification confirming HbA1c > 8.5%.",
         "rejection_reason": None, "submitted": "2025-05-24"},
        {"claim_id": "CLM-2025-0617", "member_id": "784-1978-6329401-7",
         "drug": "Zocor 20mg", "generic": "Simvastatin 20mg", "drug_class": "statin",
         "status": "rejected", "pa_required": False, "pa_reason": None,
         "rejection_reason": "Brand Zocor restricted to generic list under Daman Gold. "
                             "Please resubmit with Simvastatin 20mg (generic).",
         "submitted": "2025-05-21"},
        {"claim_id": "CLM-2025-0633", "member_id": "784-1985-7741823-5",
         "drug": "Nexium 40mg", "generic": "Esomeprazole 40mg", "drug_class": "PPI",
         "status": "rejected", "pa_required": False, "pa_reason": None,
         "rejection_reason": "Esomeprazole 80mg dose not covered under AXA Basic. "
                             "Standard 40mg covered; please adjust prescription.",
         "submitted": "2025-05-25"},
        {"claim_id": "CLM-2025-0655", "member_id": "784-1983-5524190-4",
         "drug": "Amoxil 500mg", "generic": "Amoxicillin 500mg", "drug_class": "antibiotic",
         "status": "approved", "pa_required": False,
         "pa_reason": None, "rejection_reason": None, "submitted": "2025-05-26"},
        {"claim_id": "CLM-2025-0671", "member_id": "784-1983-5524190-4",
         "drug": "Lantus", "generic": "Insulin Glargine", "drug_class": "insulin",
         "status": "under_review", "pa_required": True,
         "pa_reason": "Insulin Glargine requires PA under Daman Thiqa plan for new patients. "
                      "Submit HbA1c readings and endocrinologist letter via E-Claim.",
         "rejection_reason": None, "submitted": "2025-05-27"},
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
            {"drug": "Metformin 500mg", "tier": 1, "covered": True,  "pa_required": False},
            {"drug": "Januvia 100mg",   "tier": 3, "covered": True,  "pa_required": True},
        ],
        "biguanide": [
            {"drug": "Metformin 500mg", "tier": 1, "covered": True, "pa_required": False},
        ],
        "antiplatelet": [
            {"drug": "Aspirin 81mg",    "tier": 1, "covered": True, "pa_required": False},
            {"drug": "Ticagrelor 90mg", "tier": 3, "covered": True, "pa_required": True},
        ],
        "insulin": [
            {"drug": "Insulin Detemir", "tier": 2, "covered": True, "pa_required": False},
            {"drug": "NPH Insulin",     "tier": 1, "covered": True, "pa_required": False},
        ],
        "antibiotic": [
            {"drug": "Azithromycin 250mg",   "tier": 2, "covered": True, "pa_required": False},
            {"drug": "Clarithromycin 500mg", "tier": 2, "covered": True, "pa_required": False},
        ],
        "PPI": [
            {"drug": "Omeprazole 20mg",   "tier": 1, "covered": True, "pa_required": False},
            {"drug": "Pantoprazole 40mg", "tier": 1, "covered": True, "pa_required": False},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS_OPENAI: list[dict] = [
    {"type": "function", "function": {
        "name": "lookup_member",
        "description": "Look up member by Emirates ID. Does NOT return name.",
        "parameters": {"type": "object",
            "properties": {"emirates_id": {"type": "string"}},
            "required": ["emirates_id"]},
    }},
    {"type": "function", "function": {
        "name": "verify_member_name",
        "description": "Verify caller name matches record. Call AFTER lookup_member.",
        "parameters": {"type": "object",
            "properties": {
                "emirates_id":   {"type": "string"},
                "provided_name": {"type": "string"},
            },
            "required": ["emirates_id", "provided_name"]},
    }},
    {"type": "function", "function": {
        "name": "get_claim_status",
        "description": "Retrieve claim status for a drug and verified member.",
        "parameters": {"type": "object",
            "properties": {
                "emirates_id": {"type": "string"},
                "drug_name":   {"type": "string"},
            },
            "required": ["emirates_id", "drug_name"]},
    }},
    {"type": "function", "function": {
        "name": "get_formulary_alternatives",
        "description": "Get covered alternatives with real-time inventory.",
        "parameters": {"type": "object",
            "properties": {
                "drug_class":  {"type": "string"},
                "pharmacy_id": {"type": "string"},
            },
            "required": ["drug_class", "pharmacy_id"]},
    }},
    {"type": "function", "function": {
        "name": "get_policy_status",
        "description": "Check if member policy is active or expired.",
        "parameters": {"type": "object",
            "properties": {"emirates_id": {"type": "string"}},
            "required": ["emirates_id"]},
    }},
    {"type": "function", "function": {
        "name": "get_claim_by_id",
        "description": "Look up details of a claim by its PBM claim ID (e.g., CLM-2025-0441). Returns member ID, drug name, status, and rejection/PA reasons.",
        "parameters": {"type": "object",
            "properties": {"claim_id": {"type": "string"}},
            "required": ["claim_id"]},
    }},
]

_TOOLS_JSON_SCHEMA = json.dumps(
    [t["function"] for t in TOOLS_OPENAI], indent=2, ensure_ascii=False
)

_JSON_SHIM_ADDENDUM = f"""
You have access to the following tools. When you need to call one or more tools, output
ONLY a valid JSON array or object on a single line — no prose, no markdown fences:

  [ {{"tool": "<tool_name>", "arguments": {{...}}}}, ... ]

After receiving the tool results (injected as user messages), continue naturally
in plain text. If no tool call is needed, respond normally in plain text.

Available tools:
{_TOOLS_JSON_SCHEMA}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL EXECUTOR  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def execute_tool(name: str, inputs: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    try:
        if name == "lookup_member":
            eid = inputs.get("emirates_id", "")
            if not eid:
                return {"error": "Missing required argument: emirates_id"}, (time.perf_counter() - t0) * 1_000
            eid = eid.strip()
            m   = _DB["members"].get(eid)
            result: dict = ({"found": False} if not m else {
                "found": True, "emirates_id": m["emirates_id"],
                "policy_number": m["policy_number"], "insurer": m["insurer"],
                "plan": m["plan"], "status": m["status"],
                "copay_pct": m["copay_pct"], "expiry_date": m["expiry_date"],
                "network_pharmacy": m["network_pharmacy"],
            })
        elif name == "verify_member_name":
            eid = inputs.get("emirates_id", "")
            provided = inputs.get("provided_name", "")
            if not eid or not provided:
                missing = []
                if not eid: missing.append("emirates_id")
                if not provided: missing.append("provided_name")
                return {"error": f"Missing required argument(s): {', '.join(missing)}"}, (time.perf_counter() - t0) * 1_000
            eid = eid.strip()
            provided = provided.strip().lower()
            m = _DB["members"].get(eid)
            if not m:
                result = {"verified": False}
            else:
                stored = m["name"].strip().lower()
                result = {"verified": (provided == stored) or (provided in stored)
                                       or (stored in provided)}
        elif name == "get_claim_status":
            eid   = inputs.get("emirates_id", "")
            query = inputs.get("drug_name", "")
            if not eid or not query:
                missing = []
                if not eid: missing.append("emirates_id")
                if not query: missing.append("drug_name")
                return {"error": f"Missing required argument(s): {', '.join(missing)}"}, (time.perf_counter() - t0) * 1_000
            eid = eid.strip()
            query = query.strip().lower()
            match = next(
                (c for c in _DB["claims"]
                 if c["member_id"] == eid and (
                     query in c["drug"].lower() or query in c["generic"].lower()
                     or c["drug"].lower() in query)),
                None,
            )
            result = ({"found": False} if not match else {
                "found": True, "claim_id": match["claim_id"],
                "drug": match["drug"], "generic": match["generic"],
                "drug_class": match["drug_class"], "status": match["status"],
                "pa_required": match["pa_required"], "pa_reason": match["pa_reason"],
                "rejection_reason": match["rejection_reason"],
            })
        elif name == "get_formulary_alternatives":
            dc   = inputs.get("drug_class", "")
            pid  = inputs.get("pharmacy_id", "")
            if not dc or not pid:
                missing = []
                if not dc: missing.append("drug_class")
                if not pid: missing.append("pharmacy_id")
                return {"error": f"Missing required argument(s): {', '.join(missing)}"}, (time.perf_counter() - t0) * 1_000
            dc = dc.strip().lower()
            pid = pid.strip()
            alts = _DB["formulary_alternatives"].get(dc, [])
            inv  = _DB["inventory"].get(pid, {})
            result = {"drug_class": dc, "pharmacy_id": pid, "alternatives": [
                {**a,
                 "inventory_status": inv.get(a["drug"], {}).get("status", "unknown"),
                 "qty_on_hand":      inv.get(a["drug"], {}).get("qty", 0)}
                for a in alts
            ]}
        elif name == "get_policy_status":
            eid = inputs.get("emirates_id", "")
            if not eid:
                return {"error": "Missing required argument: emirates_id"}, (time.perf_counter() - t0) * 1_000
            eid = eid.strip()
            m   = _DB["members"].get(eid)
            result = ({"found": False} if not m else {
                "found": True, "policy_number": m["policy_number"],
                "insurer": m["insurer"], "plan": m["plan"],
                "status": m["status"], "expiry_date": m["expiry_date"],
            })
        elif name == "get_claim_by_id":
            cid = inputs.get("claim_id", "")
            if not cid:
                return {"error": "Missing required argument: claim_id"}, (time.perf_counter() - t0) * 1_000
            cid = cid.strip()
            match = next((c for c in _DB["claims"] if c["claim_id"] == cid), None)
            result = ({"found": False, "error": "Claim not found."} if not match else {
                "found": True,
                "claim_id": match["claim_id"],
                "member_id": match["member_id"],
                "drug": match["drug"],
                "generic": match["generic"],
                "drug_class": match["drug_class"],
                "status": match["status"],
                "pa_required": match["pa_required"],
                "pa_reason": match["pa_reason"],
                "rejection_reason": match["rejection_reason"],
                "submitted": match["submitted"],
            })
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        result = {"error": f"Tool execution failed: {type(e).__name__}: {e}"}
    return result, (time.perf_counter() - t0) * 1_000


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_BASE = """\
Reasoning: low
You are the NGI Pharma AI Agent — an autonomous voice agent handling inbound calls
for a Pharmacy Benefit Management (PBM) platform operated by IIRIS Health.

IDENTITY, VERIFICATION & NAVIGATION RULES
1. Authenticate before disclosing any protected information.
   - Pharmacy caller: ask for pharmacy branch ID + patient Emirates ID, then confirm name.
   - Patient caller: ask for Emirates ID and date of birth, then confirm full name.
2. If given a PBM claim number (e.g. CLM-2025-0441), use get_claim_by_id to retrieve the claim context (member ID, drug, status, etc.).
3. If the caller provides both the patient's Emirates ID and full name (along with a claim ID), call get_claim_by_id, lookup_member, and verify_member_name in parallel in a single turn to minimize latency. Disclose nothing about the claim or member until verify_member_name returns {verified: true}.
4. If verification fails: "I'm unable to verify the identity on record." Do NOT reveal stored name.

CLAIM, POLICY & MULTI-QUERY RULES
5. When a claim is "under_review" due to PA, explain why, what must be submitted, and that review takes 24-48 hours.
6. When suggesting alternatives or checking stock levels, run get_formulary_alternatives and check pharmacy inventory in parallel in a single turn.
7. If a policy is "expired", direct the caller to HR or insurer. Do not process claims.
8. Use get_policy_status or get_claim_status in parallel with other queries if checking multiple aspects of policy/status.

VOICE BEHAVIOR
- Phone call: 2-4 sentences per turn. No bullets or headers. Max 80 tokens.
- Professional, warm, efficient.
- Always use tools. Never invent data.
"""

def build_system_prompt(tool_mode: str) -> str:
    return _SYSTEM_BASE + ("\n" + _JSON_SHIM_ADDENDUM if tool_mode == "json" else "")


# ═══════════════════════════════════════════════════════════════════════════════
# JSON-SHIM PARSER  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

_JSON_TOOL_RE  = re.compile(
    r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL)
_GEMMA_TOOL_RE = re.compile(
    r'```(?:tool_code|python)?\s*\n(\w+)\(([^)]*)\)\s*\n```', re.DOTALL)

def _parse_gemma(text: str) -> dict | None:
    m = _GEMMA_TOOL_RE.search(text)
    if not m:
        return None
    args: dict = {}
    for kv in re.finditer(
            r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\S+?)(?:,|$))', m.group(2)):
        args[kv.group(1)] = kv.group(2) or kv.group(3) or kv.group(4)
    return {"name": m.group(1), "arguments": args} if m.group(1) and args else None

def parse_json_tool_calls(text: str) -> list[dict] | None:
    text = text.strip()
    # 1. Try to parse as single JSON array or object
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            tcs = []
            for item in obj:
                if isinstance(item, dict) and "tool" in item and "arguments" in item:
                    tcs.append({"name": item["tool"], "arguments": item["arguments"]})
            if tcs:
                return tcs
        elif isinstance(obj, dict):
            if "tool" in obj and "arguments" in obj:
                return [{"name": obj["tool"], "arguments": obj["arguments"]}]
    except json.JSONDecodeError:
        pass

    # 2. Try to regex find multiple single JSON tool calls in the text
    tcs = []
    for m in _JSON_TOOL_RE.finditer(text):
        try:
            tcs.append({"name": m.group(1), "arguments": json.loads(m.group(2))})
        except json.JSONDecodeError:
            pass

    # 3. Try to regex find multiple Gemma-style tool calls in the text
    for m in _GEMMA_TOOL_RE.finditer(text):
        args: dict = {}
        for kv in re.finditer(
                r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\S+?)(?:,|$))', m.group(2)):
            args[kv.group(1)] = kv.group(2) or kv.group(3) or kv.group(4)
        if m.group(1) and args:
            tcs.append({"name": m.group(1), "arguments": args})

    return tcs if tcs else None


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS  (extended with tts_overlap_ms)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TurnMetrics:
    agent_id:       int
    turn:           int
    scenario_id:    int
    stt_ms:         float = 0.0
    stt_word_count: int   = 0
    ttft_ms:        float = 0.0
    generation_ms:  float = 0.0
    llm_e2e_ms:     float = 0.0
    tokens_out:     int   = 0
    tts_first_ms:   float = 0.0   # wall-clock time from TTS call to first chunk
    tts_full_ms:    float = 0.0
    tts_chars:      int   = 0
    tts_overlap_ms: float = 0.0   # how far into LLM decode TTS first chunk arrived
    resolver_ms:    float = 0.0   # pure in-process DB lookup (sub-ms)
    routing_ms:     float = 0.0   # llm_e2e_ms − resolver_ms
    tool_name:      str   = ""
    tool_success:   bool  = True
    has_tool_call:  bool  = False

    @property
    def pipeline_ms(self) -> float:
        """Perceived silence: STT + LLM e2e + TTS first chunk (minus overlap)."""
        return self.stt_ms + self.llm_e2e_ms + max(self.tts_first_ms - self.tts_overlap_ms, 0.0)


@dataclass
class AgentMetrics:
    agent_id:            int
    scenario_id:         int
    turns:               list[TurnMetrics] = field(default_factory=list)
    scenario_duration_s: float = 0.0
    unresponded_turns:   int   = 0

    @property
    def tok_per_s(self) -> float:
        total = sum(t.tokens_out for t in self.turns)
        return total / self.scenario_duration_s if self.scenario_duration_s else 0.0

    @property
    def avg_pipeline_ms(self) -> float:
        vals = [t.pipeline_ms for t in self.turns]
        return mean(vals) if vals else 0.0

    @property
    def tool_turns(self) -> list[TurnMetrics]:
        return [t for t in self.turns if t.has_tool_call]


# ═══════════════════════════════════════════════════════════════════════════════
# ① OPTIMIZED STT: Whisper model pool
#    - float16 quantization  (~40-60% faster than float16, negligible WER delta)
#    - beam_size=1 (greedy decode) (~20-30% faster, fine for clean phone audio)
#    - VAD disabled on pre-generated clean audio (saves ~50 ms setup)
#    - Pool of WHISPER_POOL_SIZE instances to cap VRAM while serving N_AGENTS
# ═══════════════════════════════════════════════════════════════════════════════

def transcribe_wav(model: Any, wav_path: str) -> tuple[str, float]:
    """
    Transcribe a WAV file using a Whisper instance.
    Returns (transcript, elapsed_ms).
    """
    t0 = time.perf_counter()
    segments, _ = model.transcribe(
        wav_path,
        language="en",
        beam_size=1,         # greedy decode, ~25% faster on clean audio
        vad_filter=True,     # strip leading/trailing silence to prevent hallucinations
        vad_parameters=dict(min_silence_duration_ms=200),
        condition_on_previous_text=False,  # avoids hallucination carry-over
        # Domain prompt so Whisper handles Emirates IDs and drug names correctly
        initial_prompt=(
            "Pharmacy benefit management, Emirates ID, insurance claim, "
            "prior authorization, formulary, copay, Daman, ADNIC, NAS, AXA, Cigna, "
            "Metformin, Simvastatin, Atorvastatin, Januvia, Lantus, Nexium, Plavix."
        ),
    )
    transcript = " ".join(seg.text.strip() for seg in segments).strip()
    return transcript, (time.perf_counter() - t0) * 1_000


# ═══════════════════════════════════════════════════════════════════════════════
# ② OPTIMIZED TTS: Kokoro sentence-streaming pipeline
#    - Sentence-boundary streaming: TTS starts on the first complete sentence
#      that arrives from the LLM stream, hiding tts_first_ms inside llm_e2e_ms
# ═══════════════════════════════════════════════════════════════════════════════

# Sentence boundary: end with . ! ? … or — followed by space or end-of-string,
# but NOT after common abbreviations (Mr. Dr. vs. etc.).
# The negative lookbehind covers the most frequent medical/professional ones.
_SENT_END_RE = re.compile(
    r'(?<!\bMr)(?<!\bMs)(?<!\bMrs)(?<!\bDr)(?<!\bSt)(?<!\bvs)'
    r'(?<!\bNo)(?<!\betc)(?<!\b[A-Z])'   # single-letter abbreviation (e.g. "J.")
    r'[.!?…]+(?:\s|$)'
)


def _split_sentences(text: str) -> list[str]:
    """
    Split text at sentence boundaries. Returns a list of sentence strings
    (each including its trailing punctuation and any trailing whitespace).
    """
    result: list[str] = []
    last = 0
    for m in _SENT_END_RE.finditer(text):
        end = m.end()
        chunk = text[last:end]
        if chunk.strip():
            result.append(chunk)
        last = end
    tail = text[last:].strip()
    if tail:
        result.append(tail)
    return result or [text]


def synthesise_text(pipeline: Any, text: str) -> tuple[float, float, int]:
    """
    Synthesise *text* using a Kokoro pipeline instance.
    Returns (tts_first_ms, tts_full_ms, char_count).
    Used for non-streamed synthesis (warmup, tool responses, fallback).
    """
    import torch
    ac_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    t0, t_first = time.perf_counter(), None
    with torch.autocast("cuda", dtype=ac_dtype):
        for _, _, audio in pipeline(text, voice=KOKORO_VOICE):
            if audio is not None and t_first is None:
                t_first = time.perf_counter()
    t_end = time.perf_counter()
    return (
        (t_first - t0) * 1_000 if t_first else 0.0,
        (t_end   - t0) * 1_000,
        len(text),
    )


def synthesise_streaming(
    pipeline: Any,
    text_iter: "queue.Queue[str | None]",
    tts_first_event: threading.Event,
    first_chunk_ts: list[float],  # out-param: [timestamp_of_first_audio_chunk]
    abort_event: threading.Event | None = None,
) -> tuple[float, int]:
    """
    Pull sentence chunks from *text_iter* (None = done), synthesise each,
    and set *tts_first_event* when the first audio chunk is ready.
    """
    t0          = time.perf_counter()
    total_chars = 0
    buf         = ""

    import torch
    ac_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    with torch.autocast("cuda", dtype=ac_dtype):
        while True:
            # Respect abort BEFORE blocking on the queue.
            if abort_event is not None and abort_event.is_set():
                break
            try:
                chunk = text_iter.get(timeout=1.0)   # short poll so abort_event is checked
            except queue.Empty:
                if abort_event is not None and abort_event.is_set():
                    break
                # Keep waiting with another short poll
                continue
            if chunk is None:
                break

            buf        += chunk
            total_chars += len(chunk)

            # OPTIMIZATION: only run regex sentence-splitter if token contains punctuation
            if any(p in chunk for p in (".", "!", "?", "…", "—")):
                sentences   = _split_sentences(buf)
                # Keep the last incomplete sentence in the buffer
                if len(sentences) > 1 or (sentences and not _SENT_END_RE.search(buf)):
                    complete = sentences[:-1] if not _SENT_END_RE.search(buf) else sentences
                    buf      = sentences[-1]  if not _SENT_END_RE.search(buf) else ""
                else:
                    complete = sentences if _SENT_END_RE.search(buf) else []
                    buf      = "" if _SENT_END_RE.search(buf) else buf
            else:
                complete = []

            for sent in complete:
                if abort_event is not None and abort_event.is_set():
                    break   # skip remaining sentences if aborted
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) < TTS_STREAM_MIN_CHARS and not _SENT_END_RE.search(sent):
                    buf = sent + " " + buf
                    continue
                for _, _, audio in pipeline(sent, voice=KOKORO_VOICE):
                    if audio is not None and not tts_first_event.is_set():
                        first_chunk_ts.append(time.perf_counter())
                        tts_first_event.set()

        # Flush remaining buffer (only if not aborted)
        if buf.strip() and not (abort_event is not None and abort_event.is_set()):
            for _, _, audio in pipeline(buf.strip(), voice=KOKORO_VOICE):
                if audio is not None and not tts_first_event.is_set():
                    first_chunk_ts.append(time.perf_counter())
                    tts_first_event.set()

    return (time.perf_counter() - t0) * 1_000, total_chars


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-GENERATION: Kokoro TTS for caller utterances
# ═══════════════════════════════════════════════════════════════════════════════

def pregen_caller_audio(
    scenarios: dict[int, dict],
) -> dict[tuple[int, int], str]:
    """
    Synthesise every caller utterance to WAV. Reuses existing files.
    Returns {(scenario_id, turn_idx): wav_path}.
    """
    import soundfile as sf
    import numpy as np
    import torch
    from kokoro import KPipeline

    print("\n[Kokoro] Pre-generating caller audio files...")
    t0 = time.perf_counter()
    os.makedirs(AUDIO_DIR, exist_ok=True)
    wav_map: dict[tuple[int, int], str] = {}

    # Initialize a temporary KPipeline on GPU 1 (cuda:0)
    pipeline = KPipeline(lang_code="a", device="cuda:0")
    ac_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    for sid, s in scenarios.items():
        for tidx, utterance in enumerate(s["turns"], start=1):
            wav_path = os.path.join(AUDIO_DIR, f"s{sid}_t{tidx}_v2.wav")
            if not os.path.exists(wav_path):
                chunks: list[Any] = []
                with torch.autocast("cuda", dtype=ac_dtype):
                    for _, _, audio in pipeline(utterance, voice=KOKORO_VOICE):
                        if audio is not None:
                            chunks.append(audio)
                if chunks:
                    sf.write(wav_path, np.concatenate(chunks), KOKORO_SAMPLE_RATE)
            wav_map[(sid, tidx)] = wav_path

    # Clean up immediately after audio generation to free up GPU 1 VRAM
    del pipeline
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[Kokoro] Pre-generated {len(wav_map)} WAV files in "
          f"{(time.perf_counter()-t0)*1000:.0f} ms  →  {AUDIO_DIR}")
    return wav_map


# ═══════════════════════════════════════════════════════════════════════════════
# ③ OPTIMIZED STREAMING LLM CALL
#    - Mark TTFT on the very first delta (any field), not only on non-empty content
#    - Pipe text tokens into a Queue for concurrent TTS processing
#    - tool_calls_raw is reconstructed from streaming deltas as before
# ═══════════════════════════════════════════════════════════════════════════════

def _streaming_call(
    client,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    tool_choice: str | None,
    text_queue: "queue.Queue[str | None] | None" = None,
) -> tuple[str, list | None, float, float, float, int]:
    """
    Stream one LLM completion.

    If *text_queue* is provided, each text token is put() into it as it arrives
    so the TTS thread can start synthesising the first sentence immediately.
    Sentinel None is put() when the stream ends.

    Returns (full_text, tool_calls, ttft_ms, generation_ms, e2e_ms, tokens_out).
    """
    kwargs: dict[str, Any] = dict(
        model=model, messages=messages,
        temperature=0.3, max_tokens=LLM_MAX_TOKENS,
        stream=True, stream_options={"include_usage": True},
    )
    if tools:
        kwargs["tools"]       = tools
        kwargs["tool_choice"] = tool_choice or "auto"

    dispatch_ts      = time.perf_counter()
    t_first          = None
    content_buf      = []
    tool_calls_raw:  list[dict] = []
    tokens_out       = 0

    stream = client.chat.completions.create(**kwargs)

    for chunk in stream:
        # ── mark TTFT on the very first delta (any non-empty field) ──────────
        if t_first is None:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is not None:
                has_content   = bool(delta.content)
                has_tool      = bool(getattr(delta, "tool_calls", None))
                has_role      = bool(getattr(delta, "role", None))
                if has_content or has_tool or has_role:
                    t_first = time.perf_counter()

        if not chunk.choices:
            if chunk.usage:
                tokens_out = chunk.usage.completion_tokens or 0
            continue

        delta = chunk.choices[0].delta

        if delta.content:
            content_buf.append(delta.content)
            if text_queue is not None:
                text_queue.put(delta.content)   # feed TTS pipeline

        if getattr(delta, "tool_calls", None):
            for tc_d in delta.tool_calls:
                idx = tc_d.index
                while len(tool_calls_raw) <= idx:
                    tool_calls_raw.append(
                        {"id": None, "type": "function",
                         "function": {"name": "", "arguments": ""}})
                if tc_d.id:
                    tool_calls_raw[idx]["id"] = tc_d.id
                if tc_d.function:
                    if tc_d.function.name:
                        tool_calls_raw[idx]["function"]["name"] += tc_d.function.name
                    if tc_d.function.arguments:
                        tool_calls_raw[idx]["function"]["arguments"] += tc_d.function.arguments

    if text_queue is not None:
        text_queue.put(None)   # signal TTS thread that stream is done

    t_end = time.perf_counter()
    if t_first is None:
        t_first = dispatch_ts

    return (
        "".join(content_buf),
        tool_calls_raw or None,
        (t_first   - dispatch_ts) * 1_000,
        (t_end     - t_first)     * 1_000,
        (t_end     - dispatch_ts) * 1_000,
        tokens_out,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

_SCENARIOS: dict[int, dict] = {
    1: {
        "name": "Scenario 1 — Complex Pharmacy Call (Step Therapy, Alternatives & Refill Block)",
        "caller_type": "pharmacy_staff",
        "turns": [
            "Hi, I'm calling from Dubai Pharmacy branch 005. I'd like to check on claim CLM-2025-0441.",
            "Yes, the patient is Omar Ali and the Emirates ID is 784-1996-7169603-3.",
            "What are the step therapy requirements for Zocor under his plan, and do you show covered alternatives in the statin class with available inventory here at DXB-PH-005?",
            "If we switch to Atorvastatin 20mg, does his copay change under ADNIC Enhanced? Also, check if there is an approved Metformin 500mg claim (CLM-2025-0490) on file for him.",
            "Is it too early to refill that Metformin claim, and does the plan require prior authorization for his other drug Lantus? Let's check Lantus claim status for him too.",
            "Okay, I'll advise the patient on Metformin. Now, I have another claim from a different patient: CLM-2025-0617. Let's check that one.",
            "The patient is Ravi Reyes, Emirates ID 784-1978-6329401-7.",
            "What generic should we resubmit for Ravi, and do we have stock of it here at DXB-PH-005?",
            "Perfect. We'll resubmit Simvastatin 20mg for Ravi. Are there any other active claims or policy issues on file for him?",
        ],
    },
    2: {
        "name": "Scenario 2 — Patient PA Inquiry, Benefit Check & Family Member Query",
        "caller_type": "patient",
        "turns": [
            "Hello, I'm calling to check the status of my claim CLM-2025-0512.",
            "My Emirates ID is 784-2004-2137407-6 and my name is Ahmed Khan. My date of birth is March 16th, 1988.",
            "What does the PA process entail, how long does it take, and do I have any covered alternatives that don't require prior authorization under my plan?",
            "Can you check if I have a claim for Metformin already, and what the copay would be under my plan? Also check my remaining policy benefit balance.",
            "Is that Metformin claim already dispensed, and can I also check a claim status for my family member Hana Patel: CLM-2025-0601?",
            "Her Emirates ID is 784-1983-4821093-1 and her name is Hana Patel.",
            "What covered alternatives do we have for Lantus under her plan, and do you show stock for them at Dubai Pharmacy DXB-PH-022?",
            "Great. We'll speak with her doctor about switching to Insulin Detemir. Thank you for your help!",
        ],
    },
    3: {
        "name": "Scenario 3 — Expired Policy, Rejection Explanations & Drug Switches",
        "caller_type": "patient",
        "turns": [
            "Hi, I tried to fill a prescription at the pharmacy and they said it was rejected. I have claim CLM-2025-0530. Can you tell me why?",
            "My Emirates ID is 784-1974-3341057-2 and my name is Fatima Al Mansoori. My birthday is May 5th, 1982.",
            "I see. My company said they renewed it. In the meantime, is there an active claim for my stomach medication Nexium under my name?",
            "Oh, my mistake, it must be under my sister Deepa Ali's policy. Her Emirates ID is 784-1985-7741823-5. Can you check claim CLM-2025-0633 for her?",
            "Her name is Deepa Ali and she's verified this with me.",
            "What covered alternatives are available for Nexium (PPI drug class) under her plan, and is Pantoprazole in stock at pharmacy DXB-PH-029?",
            "Great, we will ask the doctor to switch Deepa to Pantoprazole 40mg. For my Plavix, is there any generic alternative like Aspirin, and what is its stock at my nearest pharmacy DXB-PH-005?",
            "Excellent. I will get that sorted with HR and get the new prescriptions. Thank you for your assistance!",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# ④ OPTIMIZED CONVERSATION RUNNER — pipelined STT → LLM(stream) → TTS
# ═══════════════════════════════════════════════════════════════════════════════

_PRINT_LOCK = threading.Lock()


def run_conversation(
    scenario_name: str,
    caller_turns:  list[str],
    caller_type:   str,
    cfg:           ModelConfig,
    base_url:      str,
    agent_id:      int,
    scenario_id:   int,
    wav_map:       dict[tuple[int, int], str],
    whisper_model: Any,
    kokoro_pipeline: Any,
) -> AgentMetrics:
    from openai import OpenAI

    client      = OpenAI(
        base_url=base_url,
        api_key="not-needed",
        timeout=120.0,
        max_retries=5,
    )
    metrics     = AgentMetrics(agent_id=agent_id, scenario_id=scenario_id)
    messages:   list[dict] = [
        {"role": "system", "content": build_system_prompt(cfg.tool_mode)}
    ]
    scenario_t0 = time.perf_counter()

    with _PRINT_LOCK:
        print(f"\n  [A{agent_id}] ── {scenario_name}  "
              f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

    for turn_idx, original_utterance in enumerate(caller_turns, start=1):
        tm = TurnMetrics(agent_id=agent_id, turn=turn_idx, scenario_id=scenario_id)

        # ── STT ───────────────────────────────────────────────────────────────
        wav_path = wav_map.get((scenario_id, turn_idx))
        if wav_path and os.path.exists(wav_path):
            transcript, stt_ms = transcribe_wav(whisper_model, wav_path)
            tm.stt_ms         = stt_ms
            tm.stt_word_count = len(transcript.split())
            utterance         = transcript if transcript else original_utterance
        else:
            utterance  = original_utterance
            tm.stt_ms  = 0.0

        with _PRINT_LOCK:
            print(f"  [A{agent_id}|T{turn_idx}] STT: {original_utterance!r} -> {utterance!r}")

        messages.append({"role": "user", "content": utterance})
        responded      = False
        call_idx       = 0
        turn_llm_t0    = time.perf_counter()
        first_ttft     = None
        total_tokens   = 0
        tts_overlap    = 0.0   # ms into llm decode when TTS first chunk was ready

        # ── Per-turn guard: one bad turn must not crash the whole conversation ──
        try:
            _prev_tcs: list | None = None   # used in native mode to skip TTS on tool-calls

            while call_idx < MAX_TOOL_LOOPS:
                call_idx += 1
                is_tool_call_iter = (_prev_tcs is not None)  # previous call was a tool call

                # ── NATIVE tool-calling with sentence-streaming TTS ───────────
                if cfg.tool_mode == "native":
                    # Only stream to TTS on the first call (when content is likely text).
                    # On subsequent calls (after tool calls) content is usually empty;
                    # skipping TTS avoids pool starvation from orphaned threads.
                    tts_abort    = threading.Event()
                    tq           = queue.Queue()   # unbounded — never blocks _streaming_call
                    tts_first_event = threading.Event()
                    first_chunk_ts: list[float] = []
                    tts_start_ts = time.perf_counter()

                    feed_tts = not is_tool_call_iter   # only pipe to TTS on first pass
                    tts_thread = threading.Thread(
                        target=synthesise_streaming,
                        args=(kokoro_pipeline, tq, tts_first_event, first_chunk_ts, tts_abort),
                        daemon=True,
                    )
                    tts_thread.start()

                    text, tcs, ttft_ms, gen_ms, e2e_ms, toks = _streaming_call(
                        client, cfg.hf_repo, messages,
                        tools=TOOLS_OPENAI, tool_choice="auto",
                        text_queue=tq if feed_tts else None,
                    )
                    if first_ttft is None:
                        first_ttft = ttft_ms
                    total_tokens += toks

                    if text.strip():
                        responded = True

                    if not tcs:
                        # Final text response — wait for TTS first chunk
                        tts_thread.join(timeout=10.0)
                        if first_chunk_ts:
                            tm.tts_first_ms = (first_chunk_ts[0] - tts_start_ts) * 1_000
                            tm.tts_overlap_ms = max(
                                (first_chunk_ts[0] - (turn_llm_t0 + e2e_ms / 1_000)) * 1_000,
                                0.0,
                            )
                        messages.append({"role": "assistant", "content": text})
                        _prev_tcs = None
                        break
                    else:
                        # Tool call — abort TTS immediately
                        tts_abort.set()
                        tq.put(None)           # wake any blocked queue.get
                        tts_thread.join(timeout=5.0)   # wait for pool slot release

                    _prev_tcs = tcs
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
                        tm.has_tool_call = True
                        tm.tool_name     = fn_name
                        tm.resolver_ms   = resolver_ms
                        tm.tool_success  = "error" not in result

                        with _PRINT_LOCK:
                            print(f"  [A{agent_id}|T{turn_idx}] TOOL {fn_name}"
                                  f"  ttft={ttft_ms:.0f}ms  res={resolver_ms:.3f}ms")

                        messages.append({
                            "role": "tool", "tool_call_id": tc["id"],
                            "content": json.dumps(result, ensure_ascii=False),
                        })

                # ── JSON-SHIM with sentence-streaming TTS ─────────────────────
                else:
                    tts_abort    = threading.Event()
                    tq           = queue.Queue()   # unbounded — never blocks _streaming_call
                    tts_first_event = threading.Event()
                    first_chunk_ts: list[float] = []
                    tts_start_ts = time.perf_counter()

                    # Don't start TTS on iterations where we expect a tool call
                    # (call_idx > 1 means we just processed a tool result).
                    feed_tts = not is_tool_call_iter
                    tts_thread = threading.Thread(
                        target=synthesise_streaming,
                        args=(kokoro_pipeline, tq, tts_first_event, first_chunk_ts, tts_abort),
                        daemon=True,
                    )
                    tts_thread.start()

                    text, _, ttft_ms, gen_ms, e2e_ms, toks = _streaming_call(
                        client, cfg.hf_repo, messages,
                        tools=None, tool_choice=None,
                        text_queue=tq if feed_tts else None,
                    )
                    if first_ttft is None:
                        first_ttft = ttft_ms
                    total_tokens += toks

                    tcs = parse_json_tool_calls(text)

                    if tcs is None:
                        # Final text response — wait for TTS first chunk
                        tts_thread.join(timeout=10.0)
                        if text.strip():
                            responded = True
                            if first_chunk_ts:
                                tm.tts_first_ms = (first_chunk_ts[0] - tts_start_ts) * 1_000
                                tm.tts_overlap_ms = max(
                                    (first_chunk_ts[0] - (turn_llm_t0 + e2e_ms / 1_000)) * 1_000,
                                    0.0,
                                )
                        else:
                            # Model returned empty text — abort TTS and flag it
                            tts_abort.set()
                            tq.put(None)
                            with _PRINT_LOCK:
                                print(f"  [A{agent_id}|T{turn_idx}] WARN empty response "
                                      f"after {call_idx} call(s)  tq_empty={tq.empty()}")
                        messages.append({"role": "assistant", "content": text})
                        _prev_tcs = None
                        break
                    else:
                        # Tool call — abort TTS immediately
                        tts_abort.set()
                        tq.put(None)           # wake any blocked queue.get
                        tts_thread.join(timeout=5.0)   # wait for pool slot release

                        _prev_tcs = tcs   # mark that next iter follows a tool call
                        messages.append({"role": "assistant", "content": text})

                        results_list = []
                        for tc in tcs:
                            fn_name = tc["name"]
                            fn_args = tc["arguments"]
                            result, resolver_ms = execute_tool(fn_name, fn_args)
                            tm.has_tool_call = True
                            tm.tool_name     = fn_name
                            tm.resolver_ms   = resolver_ms
                            tm.tool_success  = tm.tool_success and ("error" not in result)

                            with _PRINT_LOCK:
                                print(f"  [A{agent_id}|T{turn_idx}] TOOL {fn_name}({fn_args}) -> {result} | res={resolver_ms:.3f}ms")

                            results_list.append(f"[TOOL RESULT for {fn_name}]: {json.dumps(result, ensure_ascii=False)}")

                        messages.append({
                            "role": "user",
                            "content": "\n".join(results_list),
                        })

            else:
                # MAX_TOOL_LOOPS reached — force a break so turn gets recorded
                with _PRINT_LOCK:
                    print(f"  [A{agent_id}|T{turn_idx}] WARN MAX_TOOL_LOOPS "
                          f"({MAX_TOOL_LOOPS}) reached — forcing turn end")

        except Exception as exc:
            with _PRINT_LOCK:
                print(f"  [A{agent_id}|T{turn_idx}] ERROR {type(exc).__name__}: {exc}")
            # Append the partial turn metrics so it shows as unresponded
            # but does not crash the remaining turns for this agent.

        # ── Aggregate LLM timing ──────────────────────────────────────────────
        tm.llm_e2e_ms    = (time.perf_counter() - turn_llm_t0) * 1_000
        tm.ttft_ms       = first_ttft or 0.0
        tm.generation_ms = max(tm.llm_e2e_ms - tm.ttft_ms, 0.0)
        tm.tokens_out    = total_tokens
        tm.routing_ms    = tm.llm_e2e_ms - tm.resolver_ms

        if not responded:
            metrics.unresponded_turns += 1

        metrics.turns.append(tm)

        with _PRINT_LOCK:
            print(
                f"  [A{agent_id}|T{turn_idx}]"
                f"  stt={tm.stt_ms:.0f}ms"
                f"  llm={tm.llm_e2e_ms:.0f}ms"
                f"  tts1={tm.tts_first_ms:.0f}ms (overlap={tm.tts_overlap_ms:.0f}ms)"
                f"  pipeline={tm.pipeline_ms:.0f}ms"
            )

    metrics.scenario_duration_s = time.perf_counter() - scenario_t0
    return metrics





# ═══════════════════════════════════════════════════════════════════════════════
# METRICS REPORT  (extended with overlap and target check)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_stdev(vals: list[float]) -> float:
    return stdev(vals) if len(vals) >= 2 else 0.0

def _p95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    sv = sorted(vals)
    return sv[max(int(len(sv) * 0.95) - 1, 0)]


def print_metrics_report(
    cfg:          ModelConfig,
    all_metrics:  list[AgentMetrics],
    scenario_ids: list[int],
) -> None:
    W   = 78
    sep = "═" * W
    s2  = "─" * W

    print(f"\n{sep}")
    print(f"  FULL VOICE PIPELINE METRICS REPORT  (optimized)")
    print(f"  Model   : {cfg.display_name}")
    print(f"  STT     : Whisper Large V3  float16  beam=1  (dedicated per agent)")
    print(f"  TTS     : Kokoro-82M        float16/bf16  sentence-streaming  (dedicated per agent)")
    print(f"  LLM     : shared vLLM  ({N_AGENTS} KV-cache contexts)  |  warmed up before scenarios")
    print(f"  GPU     : {ACTIVE_GPU}")
    print(f"  Target  : avg e2e (pipeline_ms) < 6 000 ms per agent")
    print(f"{sep}\n")

    all_pipeline:  list[float] = []
    all_stt:       list[float] = []
    all_ttft:      list[float] = []
    all_llm_e2e:   list[float] = []
    all_tts1:      list[float] = []
    all_tts_full:  list[float] = []
    all_resolver:  list[float] = []
    all_routing:   list[float] = []
    all_toks:      list[float] = []
    all_overlap:   list[float] = []

    print(f"  {'Scen':<6} {'Ags':>3} {'tok/s':>6} {'pipeline':>10} "
          f"{'stt':>7} {'ttft':>7} {'llm':>7} {'tts1':>7} {'overlap':>8} {'routing':>9}")
    print(f"  {s2}")

    for sid in scenario_ids:
        grp = [m for m in all_metrics if m.scenario_id == sid]
        if not grp:
            continue
        pip  = [t.pipeline_ms  for m in grp for t in m.turns]
        stt  = [t.stt_ms       for m in grp for t in m.turns if t.stt_ms > 0]
        ttft = [t.ttft_ms      for m in grp for t in m.turns if t.ttft_ms > 0]
        le2e = [t.llm_e2e_ms   for m in grp for t in m.turns]
        tt1  = [t.tts_first_ms for m in grp for t in m.turns if t.tts_first_ms > 0]
        ovlp = [t.tts_overlap_ms for m in grp for t in m.turns if t.tts_overlap_ms > 0]
        rtng = [t.routing_ms     for m in grp for t in m.turns if t.has_tool_call]
        tks  = [m.tok_per_s      for m in grp]

        for lst, tgt in [(pip, all_pipeline), (stt, all_stt), (ttft, all_ttft),
                          (le2e, all_llm_e2e), (tt1, all_tts1), (ovlp, all_overlap),
                          (tks, all_toks)]:
            tgt.extend(lst)
        all_resolver.extend([t.resolver_ms for m in grp for t in m.turns if t.has_tool_call])
        all_routing.extend(rtng)

        print(
            f"  S{sid:<5} {len(grp):>3}"
            f" {mean(tks) if tks else 0:>6.1f}"
            f" {mean(pip) if pip else 0:>10.0f}"
            f" {mean(stt) if stt else 0:>7.0f}"
            f" {mean(ttft) if ttft else 0:>7.0f}"
            f" {mean(le2e) if le2e else 0:>7.0f}"
            f" {mean(tt1) if tt1 else 0:>7.0f}"
            f" {mean(ovlp) if ovlp else 0:>8.0f}"
            f" {mean(rtng) if rtng else 0:>9.0f}"
        )

    print(f"  {s2}\n")

    def row(label: str, vals: list[float], unit: str = "ms", fmt: str = ".0f") -> None:
        if not vals:
            print(f"  {label:<52}  {'—':>8}")
            return
        print(
            f"  {label:<52}  {mean(vals):>6{fmt}} {unit}"
            f"  σ={_safe_stdev(vals):>5.0f}"
            f"  p50={median(vals):>6.0f}"
            f"  p95={_p95(vals):>6.0f}"
        )

    print(f"  {'METRIC':<52}  {'MEAN':>8}   σ       p50      p95")
    print(f"  {s2}")
    row("PERCEIVED SILENCE  (stt + llm + tts_first − overlap)", all_pipeline)

    # ── 6-second target check ────────────────────────────────────────────────
    avg_pip = mean(all_pipeline) if all_pipeline else 0.0
    pct_over = sum(1 for v in all_pipeline if v >= 6_000) / max(len(all_pipeline), 1) * 100
    target_ok = "✓  PASS" if avg_pip < 6_000 else "✗  FAIL"
    print(f"  {'  ↳ 6 000 ms target':<52}  {avg_pip:>6.0f} ms  {target_ok}  ({pct_over:.0f}% turns over)")
    print(f"  {s2}")

    row("  STT  Whisper LV3 float16 beam=1",              all_stt)
    row("  LLM  time-to-first-token (TTFT)",                   all_ttft)
    row("  LLM  generation (decode)",
        [t.generation_ms for m in all_metrics for t in m.turns])
    row("  LLM  full round-trip e2e",                          all_llm_e2e)
    row("  TTS  Kokoro first audio chunk (wall-clock)",        all_tts1)
    row("  TTS  pipeline overlap (hidden by LLM decode)",      all_overlap)
    print(f"  {s2}")
    row("  Tool resolver latency (DB only)",                   all_resolver, "ms", ".3f")
    row("  Tool routing latency  (llm_e2e − resolver)",        all_routing)
    print(f"  {s2}")
    row("  tok/s (LLM generation throughput)",                 all_toks, "tok/s", ".1f")

    tool_turns  = [t for m in all_metrics for t in m.turns if t.has_tool_call]
    total_turns = sum(len(m.turns) for m in all_metrics)
    ok_tools    = sum(1 for t in tool_turns if t.tool_success)
    failed_tools = len(tool_turns) - ok_tools

    print(f"\n  Total conversational turns                             {total_turns:>6}")
    print(f"  Turns with tool calls                                  {len(tool_turns):>6}")
    print(f"    ↳ Successful                                         {ok_tools:>6}")
    print(f"    ↳ Failed                                             {failed_tools:>6}")
    print(f"    ↳ Success rate                                       "
          f"{ok_tools/len(tool_turns)*100 if tool_turns else 0:>5.1f} %")
    print(f"  Unresponded turns                                      "
          f"{sum(m.unresponded_turns for m in all_metrics):>6}")
    print(f"\n{sep}\n")

    print(f"  PER-AGENT SUMMARY")
    print(f"  {'Ag':<3} {'S':>2} {'tok/s':>6} {'pipeline':>10} "
          f"{'stt':>7} {'llm':>7} {'tts1':>7} {'overlap':>8} {'tools':>6} {'ok':>4} {'unrsp':>6}")
    print(f"  {s2}")
    for m in sorted(all_metrics, key=lambda x: (x.scenario_id, x.agent_id)):
        pip  = [t.pipeline_ms    for t in m.turns]
        stt  = [t.stt_ms         for t in m.turns if t.stt_ms > 0]
        le2e = [t.llm_e2e_ms     for t in m.turns]
        tt1  = [t.tts_first_ms   for t in m.turns if t.tts_first_ms > 0]
        ovlp = [t.tts_overlap_ms for t in m.turns if t.tts_overlap_ms > 0]
        n_tool = len(m.tool_turns)
        n_ok   = sum(1 for t in m.tool_turns if t.tool_success)
        print(
            f"  A{m.agent_id:<2} {m.scenario_id:>2}"
            f" {m.tok_per_s:>6.1f}"
            f" {mean(pip) if pip else 0:>10.0f}"
            f" {mean(stt) if stt else 0:>7.0f}"
            f" {mean(le2e) if le2e else 0:>7.0f}"
            f" {mean(tt1) if tt1 else 0:>7.0f}"
            f" {mean(ovlp) if ovlp else 0:>8.0f}"
            f" {n_tool:>6}"
            f" {n_ok:>4}"
            f" {m.unresponded_turns:>6}"
        )
    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS SERIALISER
# ═══════════════════════════════════════════════════════════════════════════════

def _dump_results(
    cfg:          ModelConfig,
    all_metrics:  list[AgentMetrics],
    scenario_ids: list[int],
    run_ts:       str,
) -> None:
    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": 0, "stdev": 0, "p50": 0, "p95": 0, "n": 0}
        sv = sorted(vals)
        return {
            "mean":  round(mean(vals), 3),
            "stdev": round(_safe_stdev(vals), 3),
            "p50":   round(median(vals), 3),
            "p95":   round(sv[max(int(len(sv)*0.95)-1, 0)], 3),
            "n":     len(vals),
        }

    all_turns = [t for m in all_metrics for t in m.turns]
    all_tool_turns = [t for t in all_turns if t.has_tool_call]
    payload = {
        "run_timestamp":      run_ts,
        "active_gpu":         ACTIVE_GPU,
        "model_key":          cfg.key,
        "display_name":       cfg.display_name,
        "stt_model":          "Whisper Large V3 float16 beam=1 (dedicated per agent)",
        "tts_model":          "Kokoro-82M float16/bf16 sentence-streaming (dedicated per agent)",
        "n_agents":           N_AGENTS,
        "tensor_parallel":    cfg.tensor_parallel,
        "data_parallel":      cfg.data_parallel,
        "llm_max_tokens":     LLM_MAX_TOKENS,
        "tts_stream_min_chars": TTS_STREAM_MIN_CHARS,
        "target_6s_pass":     mean([t.pipeline_ms for t in all_turns]) < 6_000 if all_turns else False,
        "summary": {
            "perceived_silence_ms": _stats([t.pipeline_ms for t in all_turns]),
            "stt_ms":               _stats([t.stt_ms for t in all_turns if t.stt_ms > 0]),
            "ttft_ms":              _stats([t.ttft_ms for t in all_turns if t.ttft_ms > 0]),
            "llm_e2e_ms":           _stats([t.llm_e2e_ms for t in all_turns]),
            "tts_first_ms":         _stats([t.tts_first_ms for t in all_turns if t.tts_first_ms > 0]),
            "tts_overlap_ms":       _stats([t.tts_overlap_ms for t in all_turns if t.tts_overlap_ms > 0]),
            "resolver_ms":          _stats([t.resolver_ms for t in all_turns if t.has_tool_call]),
            "routing_ms":           _stats([t.routing_ms for t in all_turns if t.has_tool_call]),
            "tok_per_s":            _stats([m.tok_per_s for m in all_metrics]),
            "total_turns":          len(all_turns),
            "tool_turns":           len(all_tool_turns),
            "tool_success_count":   sum(1 for t in all_tool_turns if t.tool_success),
            "tool_failure_count":   sum(1 for t in all_tool_turns if not t.tool_success),
            "tool_success_rate":    (
                sum(1 for t in all_tool_turns if t.tool_success)
                / max(len(all_tool_turns), 1) * 100
            ),
            "unresponded_turns":   sum(m.unresponded_turns for m in all_metrics),
        },
        "per_agent": [
            {
                "agent_id": m.agent_id, "scenario_id": m.scenario_id,
                "tok_per_s": round(m.tok_per_s, 2),
                "avg_pipeline_ms": round(m.avg_pipeline_ms, 1),
                "scenario_duration_s": round(m.scenario_duration_s, 2),
                "turns": [
                    {
                        "turn":           t.turn,
                        "stt_ms":         round(t.stt_ms, 1),
                        "ttft_ms":        round(t.ttft_ms, 1),
                        "llm_e2e_ms":     round(t.llm_e2e_ms, 1),
                        "tts_first_ms":   round(t.tts_first_ms, 1),
                        "tts_overlap_ms": round(t.tts_overlap_ms, 1),
                        "tts_full_ms":    round(t.tts_full_ms, 1),
                        "pipeline_ms":    round(t.pipeline_ms, 1),
                        "tokens_out":     t.tokens_out,
                        "has_tool_call":  t.has_tool_call,
                        "tool_name":      t.tool_name,
                        "tool_success":   t.tool_success if t.has_tool_call else None,
                        "resolver_ms":    round(t.resolver_ms, 4),
                        "routing_ms":     round(t.routing_ms, 1),
                    }
                    for t in m.turns
                ],
            }
            for m in sorted(all_metrics, key=lambda x: (x.scenario_id, x.agent_id))
        ],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts_tag = run_ts.replace(":", "").replace(" ", "_").replace("-", "")
    fpath  = os.path.join(RESULTS_DIR, f"voice_{cfg.key}_{ACTIVE_GPU}_{ts_tag}.json")
    with open(fpath, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[results] → {fpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# VLLM SERVER
# ═══════════════════════════════════════════════════════════════════════════════

_NOISE_RE = re.compile(
    r"(Avg prompt throughput:|Avg generation throughput:|Running: \d+ reqs"
    r"|Waiting: \d+ reqs|GPU KV cache usage:|Prefix cache hit|HTTP/1\.1\" 200)")

def _pipe_filter(src: io.RawIOBase, dst: Any) -> None:
    try:
        for raw in src:
            if not _NOISE_RE.search(raw.decode(errors="replace")):
                dst.write(raw); dst.flush()
    except Exception:
        pass

def _start_vllm(cfg: ModelConfig, port: int) -> subprocess.Popen:
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  cfg.hf_repo,
        "--download-dir",           MODEL_CACHE,
        "--dtype",                  cfg.dtype,
        "--tensor-parallel-size",   str(cfg.tensor_parallel),
        "--max-model-len",          str(cfg.max_model_len),
        "--port",                   str(port),
        "--no-enable-log-requests",
        "--trust-remote-code",
        "--hf-token",               os.environ.get("HF_TOKEN", ""),
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs",           "32",
    ]
    if cfg.data_parallel > 1:
        # pipeline-parallel-size replicates the model across DP GPUs.
        # Each replica handles a disjoint subset of concurrent sequences,
        # halving KV-cache pressure per replica at the cost of doubled GPU count.
        cmd += ["--pipeline-parallel-size", str(cfg.data_parallel)]
    if cfg.tool_mode == "native":
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", cfg.tool_call_parser]
    if cfg.quantization:
        cmd += ["--quantization", cfg.quantization]
    cmd += cfg.extra_vllm_args

    env = os.environ.copy()
    # Bind vLLM to GPU 2 (and any subsequent GPUs if TP > 1)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(1, cfg.tensor_parallel + 1))
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_DISABLE_FLASHINFER"] = "1"
    env["VLLM_USE_V1"] = "0"
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
            print(f"[vLLM] Ready on :{port}")
            return
        except (urllib.error.URLError, ConnectionRefusedError):
            time.sleep(3)
    raise RuntimeError(f"vLLM not ready within {timeout_s}s")


# ═══════════════════════════════════════════════════════════════════════════════
# BODY
# ═══════════════════════════════════════════════════════════════════════════════

def _warmup_models(
    cfg:          ModelConfig,
    base_url:     str,
    whisper_model: Any,
    kokoro_pipeline: Any,
    wav_map:      dict[tuple[int, int], str],
) -> None:
    """
    Warm up local models inside the child process.
    """
    from openai import OpenAI
    import soundfile as sf
    import numpy as np
    import torch

    # Whisper warmup
    first_wav = next(iter(wav_map.values()), None)
    if first_wav and os.path.exists(first_wav):
        transcribe_wav(whisper_model, first_wav)
    else:
        # Fallback synthetic WAV warmup
        _wav_tmp = os.path.join(AUDIO_DIR, "_warmup.wav")
        os.makedirs(AUDIO_DIR, exist_ok=True)
        ac_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast("cuda", dtype=ac_dtype):
            _chunks = [a for _, _, a in kokoro_pipeline("Warmup.", voice=KOKORO_VOICE) if a is not None]
        if _chunks:
            sf.write(_wav_tmp, np.concatenate(_chunks), KOKORO_SAMPLE_RATE)
        if os.path.exists(_wav_tmp):
            transcribe_wav(whisper_model, _wav_tmp)

    # Kokoro warmup
    ac_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast("cuda", dtype=ac_dtype):
        for _, _, _ in kokoro_pipeline("Warmup.", voice=KOKORO_VOICE):
            pass

    # vLLM warmup
    client = OpenAI(base_url=base_url, api_key="not-needed")
    client.chat.completions.create(
        model=cfg.hf_repo,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1,
        temperature=0.0,
    )


def run_conversation_process(
    scenario_name: str,
    caller_turns: list[str],
    caller_type: str,
    cfg: ModelConfig,
    base_url: str,
    agent_id: int,
    scenario_id: int,
    wav_map: dict[tuple[int, int], str],
    result_queue: Any,
):
    """
    Worker function spawned as a separate OS process.
    Loads dedicated Whisper & Kokoro models, warms up, and runs conversation.
    """
    import os
    import torch
    from faster_whisper import WhisperModel
    from kokoro import KPipeline

    print(f"  [A{agent_id}] Starting agent process (PID: {os.getpid()})...")

    # 1. Load Whisper Large V3 on cuda:0
    whisper_model = WhisperModel(
        "large-v3",
        device="cuda",
        device_index=0,
        compute_type="float16",
        download_root=MODEL_CACHE,
        num_workers=1,
    )

    # 2. Load Kokoro on cuda:0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    kokoro_pipeline = KPipeline(lang_code="a", device=device)

    # 3. Warm up local models inside this process
    _warmup_models(cfg, base_url, whisper_model, kokoro_pipeline, wav_map)
    print(f"  [A{agent_id}] Models loaded and warmed up.")

    # 4. Run the conversation using our local models
    metrics = run_conversation(
        scenario_name=scenario_name,
        caller_turns=caller_turns,
        caller_type=caller_type,
        cfg=cfg,
        base_url=base_url,
        agent_id=agent_id,
        scenario_id=scenario_id,
        wav_map=wav_map,
        whisper_model=whisper_model,
        kokoro_pipeline=kokoro_pipeline,
    )

    # 5. Send metrics back to parent process
    result_queue.put(metrics)
    print(f"  [A{agent_id}] Finished and sent metrics.")


def _body(cfg: ModelConfig, scenario_ids: list[int]) -> None:
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Phase 1: Pre-generate caller audio ───────────────────────────────────
    # Uses a self-cleaning pipeline locally inside this parent process
    wav_map = pregen_caller_audio(
        {sid: _SCENARIOS[sid] for sid in scenario_ids}
    )

    # ── Phase 2: Start vLLM server ────────────────────────────────────────────
    proc = _start_vllm(cfg, VLLM_PORT)
    try:
        _wait_for_vllm(VLLM_PORT)
        base_url     = f"http://localhost:{VLLM_PORT}/v1"

        all_metrics: list[AgentMetrics] = []

        import multiprocessing
        # Ensure spawn start method is used for safe CUDA initialization in children
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        for sid in scenario_ids:
            s = _SCENARIOS[sid]
            print(f"\n{'─'*72}")
            print(f"  [{s['name']}]  {N_AGENTS} agents  |  STT+LLM(stream)+TTS(pipeline)")
            print(f"{'─'*72}")

            result_queue = multiprocessing.Queue()
            processes: list[multiprocessing.Process] = []

            for aid in range(1, N_AGENTS + 1):
                p = multiprocessing.Process(
                    target=run_conversation_process,
                    kwargs=dict(
                        scenario_name=s["name"],
                        caller_turns=s["turns"],
                        caller_type=s["caller_type"],
                        cfg=cfg,
                        base_url=base_url,
                        agent_id=aid,
                        scenario_id=sid,
                        wav_map=wav_map,
                        result_queue=result_queue,
                    ),
                    daemon=True,
                )
                processes.append(p)

            # Stagger by 0.5s to avoid thundering-herd on the vLLM scheduler at turn 1
            for p in processes:
                p.start()
                time.sleep(0.5)

            # Wait for all processes to finish
            for p in processes:
                p.join()

            # Retrieve metrics from queue
            batch = []
            while not result_queue.empty():
                batch.append(result_queue.get())

            all_metrics.extend(batch)

        print_metrics_report(cfg, all_metrics, scenario_ids)
        _dump_results(cfg, all_metrics, scenario_ids, run_ts)

    finally:
        proc.terminate()
        proc.wait()


# ═══════════════════════════════════════════════════════════════════════════════
# MODAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_SEC = [modal.Secret.from_name("huggingface-secret")]
_VOL = {MODEL_CACHE: model_volume, RESULTS_DIR: results_volume}


@app.function(image=image, gpu=_modal_gpu("gpt120b_mxfp4"), volumes=_VOL, timeout=7200, secrets=_SEC)
def run_gpt120b_mxfp4(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gpt120b_mxfp4"], scenario_ids)

@app.function(image=image, gpu=_modal_gpu("gpt120b_bf16"), volumes=_VOL, timeout=7200, secrets=_SEC)
def run_gpt120b_bf16(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gpt120b_bf16"], scenario_ids)

@app.function(image=image, gpu=_modal_gpu("gpt20b_mxfp4"), volumes=_VOL, timeout=7200, secrets=_SEC)
def run_gpt20b_mxfp4(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gpt20b_mxfp4"], scenario_ids)

@app.function(image=image, gpu=_modal_gpu("gpt20b_bf16"), volumes=_VOL, timeout=7200, secrets=_SEC)
def run_gpt20b_bf16(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gpt20b_bf16"], scenario_ids)

@app.function(image=image, gpu=_modal_gpu("gemma4_26b"), volumes=_VOL, timeout=7200, secrets=_SEC)
def run_gemma4_26b(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gemma4_26b"], scenario_ids)

@app.function(image=image, gpu=_modal_gpu("gemma4_31b"), volumes=_VOL, timeout=7200, secrets=_SEC)
def run_gemma4_31b(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gemma4_31b"], scenario_ids)

@app.function(image=image, gpu=_modal_gpu("qwen3_72b_fp8"), volumes=_VOL, timeout=7200, secrets=_SEC)
def run_qwen3_72b_fp8(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["qwen3_72b_fp8"], scenario_ids)

@app.function(image=image, gpu=_modal_gpu("qwen3_72b_bf16"), volumes=_VOL, timeout=7200, secrets=_SEC)
def run_qwen3_72b_bf16(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["qwen3_72b_bf16"], scenario_ids)


_RUNNERS: dict[str, Any] = {
    "gpt120b_mxfp4":  run_gpt120b_mxfp4,
    "gpt120b_bf16":   run_gpt120b_bf16,
    "gpt20b_mxfp4":   run_gpt20b_mxfp4,
    "gpt20b_bf16":    run_gpt20b_bf16,
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
    modal run on_prem.py
    modal run on_prem.py --model gemma4_26b
    modal run on_prem.py --model gpt120b_bf16 --scenario 1
    ACTIVE_GPU=B200         modal run on_prem.py
    N_AGENTS=10             modal run on_prem.py --model gemma4_26b
    WHISPER_POOL_SIZE=3     modal run on_prem.py   # more STT concurrency
    TTS_STREAM_MIN_CHARS=40 modal run on_prem.py   # lower first-audio latency
    """
    if model and model not in _RUNNERS:
        raise ValueError(f"Unknown model. Choose from: {', '.join(sorted(_RUNNERS))}")

    target_models    = [model] if model else sorted(_RUNNERS)
    all_scenario_ids = sorted(_SCENARIOS)
    target_scenarios = [scenario] if scenario in all_scenario_ids else all_scenario_ids

    W = 78
    print(f"\n{'═'*W}")
    print(f"  NGI Pharma — Full Voice Pipeline Benchmark  (optimized)")
    print(f"  GPU       : {ACTIVE_GPU}  ({_MODAL_GPU_TAG[ACTIVE_GPU]})")
    print(f"  STT       : Whisper LV3 float16  beam=1  pool={WHISPER_POOL_SIZE}")
    print(f"  TTS       : Kokoro-82M float16/bf16  sentence-streaming  pool={KOKORO_POOL_SIZE}")
    print(f"  LLM       : shared vLLM  |  {N_AGENTS} concurrent KV-cache contexts  (warmed up)")
    print(f"  Target    : avg pipeline < 6 000 ms per agent")
    print(f"  Models    : {target_models}")
    print(f"  Scenarios : {target_scenarios}")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*W}\n")

    for mk in target_models:
        cfg = MODEL_REGISTRY[mk]
        print(f"\n{'─'*W}")
        print(f"  {cfg.display_name}")
        print(f"  dtype={cfg.dtype}"
              + (f"  quant={cfg.quantization}" if cfg.quantization else "")
              + f"  tool_mode={cfg.tool_mode}")
        print(f"{'─'*W}\n")
        _RUNNERS[mk].remote(target_scenarios)