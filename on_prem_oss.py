"""
on_prem_oss.py
═══════════════════════════════════════════════════════════════════════════════
NGI Pharma Voice AI — Modal demo runner for three scripted PBM call scenarios.

Runs all three scenarios against each open-source LLM candidate using vLLM's
OpenAI-compatible server.  Each model gets its own Modal function with the
correct number of RTX PRO 6000 GPUs.  Per-tool latency is logged independently
of LLM generation time.

── GPU Allocation (RTX PRO 6000 — 96 GB GDDR7, 1.8 TB/s) ────────────────────
  GPT OSS 120B  MXFP4   ~60 GB active  → 1× RTX-PRO-6000   (TP=1)
  GPT OSS 120B  BF16    ~240 GB        → 3× RTX-PRO-6000   (TP=3, PCIe Gen5)
  Gemma 4 26B   BF16    ~63.8 GB       → 1× RTX-PRO-6000
  Gemma 4 31B   BF16    ~75.5 GB       → 1× RTX-PRO-6000
  Qwen3 72B     FP8     ~89 GB         → 1× RTX-PRO-6000
  Qwen3 72B     BF16    ~167.5 GB      → 2× RTX-PRO-6000   (TP=2, PCIe Gen5)

── Setup ──────────────────────────────────────────────────────────────────────
    modal run on_prem_oss.py                                 # all models, all scenarios
    modal run on_prem_oss.py --model gemma4_26b              # one model, all scenarios
    modal run on_prem_oss.py --model qwen3_72b_fp8 --scenario 2
    modal run on_prem_oss.py --scenario 3                    # all models, scenario 3

    Available --model keys:
      gpt120b_mxfp4   GPT OSS 120B MXFP4 — 1× RTX PRO 6000
      gpt120b_bf16    GPT OSS 120B BF16  — 3× RTX PRO 6000
      gemma4_26b      Gemma 4 26B BF16   — 1× RTX PRO 6000
      gemma4_31b      Gemma 4 31B BF16   — 1× RTX PRO 6000
      qwen3_72b_fp8   Qwen3 72B FP8      — 1× RTX PRO 6000
      qwen3_72b_bf16  Qwen3 72B BF16     — 2× RTX PRO 6000
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import os

import modal

# ─────────────────────────────────────────────────────────────────────────────
# Modal app + image
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("pharma-agent-oss-demo")

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

# Persistent volume — model weights are cached here and reused across runs.
model_volume = modal.Volume.from_name("pharma-model-weights", create_if_missing=True)
MODEL_CACHE_PATH = "/model-cache"

VLLM_PORT = 8100


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    key: str
    hf_repo: str
    display_name: str

    # GPUs required PER REPLICA
    gpu_count: int

    # number of live replicas
    replicas: int

    dtype: str
    quantization: str | None
    tensor_parallel: int
    max_model_len: int

    # estimated VRAM PER REPLICA
    vram_gb: float

    tool_mode: str
    tool_call_parser: str = "hermes"
    extra_vllm_args: list[str] = field(default_factory=list)


MODEL_REGISTRY: dict[str, ModelConfig] = {

    "gpt120b_mxfp4": ModelConfig(
        key="gpt120b_mxfp4",
        hf_repo="openai/gpt-oss-120b",
        display_name="GPT OSS 120B MXFP4",

        gpu_count=1,
        replicas=5,

        dtype="bfloat16",
        quantization="mxfp4",
        tensor_parallel=1,

        max_model_len=32768,

        # ~60-64 GB active
        vram_gb=63.8,

        tool_mode="json",

        extra_vllm_args=[
            "--enable-chunked-prefill",
        ],
    ),

    "gpt120b_bf16": ModelConfig(
        key="gpt120b_bf16",
        hf_repo="openai/gpt-oss-120b",
        display_name="GPT OSS 120B BF16",

        # 4 GPUs REQUIRED per replica
        gpu_count=4,
        replicas=5,

        dtype="bfloat16",
        quantization=None,

        tensor_parallel=4,

        max_model_len=32768,

        # ~240 GB model footprint
        vram_gb=240.0,

        tool_mode="json",

        extra_vllm_args=[
            "--enable-chunked-prefill",
        ],
    ),

    "gemma4_26b": ModelConfig(
        key="gemma4_26b",
        hf_repo="google/gemma-3-27b-it",
        display_name="Gemma 4 26B BF16",

        gpu_count=1,
        replicas=5,

        dtype="bfloat16",
        quantization=None,

        tensor_parallel=1,

        max_model_len=32768,

        vram_gb=63.8,

        tool_mode="json",
    ),

    "gemma4_31b": ModelConfig(
        key="gemma4_31b",
        hf_repo="google/gemma-3-27b-it",
        display_name="Gemma 4 31B BF16",

        gpu_count=1,
        replicas=5,

        dtype="bfloat16",
        quantization=None,

        tensor_parallel=1,

        max_model_len=32768,

        # larger KV/cache footprint
        vram_gb=75.5,

        tool_mode="json",
    ),

    "qwen3_72b_fp8": ModelConfig(
        key="qwen3_72b_fp8",
        hf_repo="Qwen/Qwen2.5-72B-Instruct",
        display_name="Qwen3 72B FP8",

        gpu_count=1,
        replicas=5,

        dtype="bfloat16",
        quantization="fp8",

        tensor_parallel=1,

        max_model_len=32768,

        vram_gb=89.0,

        tool_mode="native",
    ),

    "qwen3_72b_bf16": ModelConfig(
        key="qwen3_72b_bf16",
        hf_repo="Qwen/Qwen2.5-72B-Instruct",
        display_name="Qwen3 72B BF16",

        gpu_count=2,
        replicas=5,

        dtype="bfloat16",
        quantization=None,

        tensor_parallel=2,

        max_model_len=32768,

        vram_gb=167.5,

        tool_mode="native",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# FAKE IN-MEMORY DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

_DB: dict[str, Any] = {
    "members": {
        "784-1996-7169603-3": {
            "emirates_id":      "784-1996-7169603-3",
            "name":             "Omar Ali",
            "dob":              "1996-05-15",
            "policy_number":    "ADNIC-ENH-001",
            "insurer":          "ADNIC",
            "plan":             "ADNIC Enhanced",
            "status":           "active",
            "copay_pct":        20,
            "network_pharmacy": "DXB-PH-005",
            "expiry_date":      None,
        },
        "784-2004-2137407-6": {
            "emirates_id":      "784-2004-2137407-6",
            "name":             "Ahmed Khan",
            "dob":              "1988-03-16",
            "policy_number":    "NAS-ENH-042",
            "insurer":          "NAS",
            "plan":             "NAS Enhanced",
            "status":           "active",
            "copay_pct":        10,
            "network_pharmacy": None,
            "expiry_date":      None,
        },
        "784-1974-3341057-2": {
            "emirates_id":      "784-1974-3341057-2",
            "name":             "Fatima Al Mansoori",
            "dob":              "1982-05-05",
            "policy_number":    "CIGNA-ME-117",
            "insurer":          "Cigna ME",
            "plan":             "Cigna ME Standard",
            "status":           "expired",
            "copay_pct":        0,
            "network_pharmacy": None,
            "expiry_date":      "2024-12-16",
        },
    },
    "claims": [
        {
            "claim_id":         "CLM-2025-0441",
            "member_id":        "784-1996-7169603-3",
            "drug":             "Zocor 40mg",
            "generic":          "Simvastatin 40mg",
            "drug_class":       "statin",
            "status":           "under_review",
            "pa_required":      True,
            "pa_reason": (
                "Step therapy applies — documentation of prior failed therapy "
                "with Simvastatin or Lovastatin required before this brand is approved."
            ),
            "rejection_reason": None,
            "submitted":        "2025-05-20",
        },
        {
            "claim_id":         "CLM-2025-0512",
            "member_id":        "784-2004-2137407-6",
            "drug":             "Januvia 100mg",
            "generic":          "Sitagliptin 100mg",
            "drug_class":       "DPP-4 inhibitor",
            "status":           "under_review",
            "pa_required":      True,
            "pa_reason": (
                "Prior Authorization required per NAS formulary Tier 3 policy. "
                "Physician must submit PA form with clinical notes via E-Claim portal."
            ),
            "rejection_reason": None,
            "submitted":        "2025-05-22",
        },
        {
            "claim_id":         "CLM-2025-0530",
            "member_id":        "784-1974-3341057-2",
            "drug":             "Plavix",
            "generic":          "Clopidogrel 75mg",
            "drug_class":       "antiplatelet",
            "status":           "rejected",
            "pa_required":      False,
            "pa_reason":        None,
            "rejection_reason": "Policy expired on 2024-12-16; no active coverage.",
            "submitted":        "2025-05-23",
        },
    ],
    "inventory": {
        "DXB-PH-005": {
            "Atorvastatin 20mg": {"qty": 240, "status": "in_stock"},
            "Rosuvastatin 10mg": {"qty": 18,  "status": "low_stock"},
            "Zocor 40mg":        {"qty": 45,  "status": "in_stock"},
            "Januvia 100mg":     {"qty": 90,  "status": "in_stock"},
            "Metformin 500mg":   {"qty": 300, "status": "in_stock"},
            "Plavix":            {"qty": 60,  "status": "in_stock"},
        },
    },
    "formulary_alternatives": {
        "statin": [
            {"drug": "Atorvastatin 20mg", "tier": 1, "covered": True, "pa_required": False},
            {"drug": "Rosuvastatin 10mg",  "tier": 2, "covered": True, "pa_required": False},
        ],
        "DPP-4 inhibitor": [
            {"drug": "Metformin 500mg", "tier": 1, "covered": True, "pa_required": False},
        ],
        "antiplatelet": [
            {"drug": "Aspirin 81mg", "tier": 1, "covered": True, "pa_required": False},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS  (OpenAI-compatible schema for vLLM native tool-calling)
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS_OPENAI: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_member",
            "description": (
                "Look up an insurance member by Emirates ID. "
                "Returns policy metadata and status. "
                "Does NOT return the member name — caller must confirm via "
                "verify_member_name before any protected information is disclosed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emirates_id": {
                        "type": "string",
                        "description": "Emirates ID in canonical format, e.g. 784-1996-7169603-3",
                    },
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
                "Verify the name stated by the caller matches the record. "
                "Must be called AFTER lookup_member. "
                "Returns {verified: true/false}. "
                "Do NOT reveal the stored name if verification fails."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emirates_id": {"type": "string"},
                    "provided_name": {
                        "type": "string",
                        "description": "Name exactly as spoken by the caller.",
                    },
                },
                "required": ["emirates_id", "provided_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_claim_status",
            "description": (
                "Retrieve the current claim status for a specific drug and verified member. "
                "Only callable after identity is verified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emirates_id": {"type": "string"},
                    "drug_name": {
                        "type": "string",
                        "description": "Drug name as mentioned by the caller, e.g. 'Zocor 40mg'.",
                    },
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
                "Get covered formulary alternatives for a given drug class, with "
                "real-time inventory levels at a specific pharmacy. "
                "Use the drug_class returned by get_claim_status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_class": {
                        "type": "string",
                        "description": "Drug class, e.g. 'statin', 'DPP-4 inhibitor'.",
                    },
                    "pharmacy_id": {
                        "type": "string",
                        "description": "Pharmacy branch ID, e.g. 'DXB-PH-005'.",
                    },
                },
                "required": ["drug_class", "pharmacy_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_policy_status",
            "description": (
                "Check whether a member's insurance policy is active or expired. "
                "Use as a fast-path check when processing a new claim inquiry."
            ),
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

# JSON-mode shim: tool schemas injected into the system prompt for models that
# don't support native structured tool-call output (MoE / Mixtral-class).
_TOOLS_JSON_SCHEMA = json.dumps(
    [t["function"] for t in TOOLS_OPENAI],
    indent=2,
    ensure_ascii=False,
)

_JSON_SHIM_ADDENDUM = f"""
You have access to the following tools. When you need to call a tool, output
ONLY a valid JSON object on a single line — no prose, no markdown fences:

  {{"tool": "<tool_name>", "arguments": {{...}}}}

After you receive the tool result (injected as a user message), continue
the conversation naturally in plain text.  If no tool call is needed,
respond normally in plain text.

Available tools:
{_TOOLS_JSON_SCHEMA}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

def execute_tool(name: str, inputs: dict) -> tuple[dict, float]:
    """Execute a named tool against _DB. Returns (result, elapsed_ms)."""
    t0 = time.perf_counter()

    if name == "lookup_member":
        eid = inputs["emirates_id"].strip()
        member = _DB["members"].get(eid)
        if not member:
            result: dict = {"found": False, "error": "No member record for that Emirates ID."}
        else:
            result = {
                "found":            True,
                "emirates_id":      member["emirates_id"],
                "policy_number":    member["policy_number"],
                "insurer":          member["insurer"],
                "plan":             member["plan"],
                "status":           member["status"],
                "copay_pct":        member["copay_pct"],
                "expiry_date":      member["expiry_date"],
                "network_pharmacy": member["network_pharmacy"],
                # name intentionally omitted — must be verified separately
            }

    elif name == "verify_member_name":
        eid      = inputs["emirates_id"].strip()
        provided = inputs["provided_name"].strip().lower()
        member   = _DB["members"].get(eid)
        if not member:
            result = {"verified": False, "reason": "Member not found."}
        else:
            stored = member["name"].strip().lower()
            match  = (provided == stored) or (provided in stored) or (stored in provided)
            result = {"verified": match}

    elif name == "get_claim_status":
        eid   = inputs["emirates_id"].strip()
        query = inputs["drug_name"].strip().lower()
        matched = None
        for claim in _DB["claims"]:
            if claim["member_id"] == eid and (
                query in claim["drug"].lower()
                or query in claim["generic"].lower()
                or claim["drug"].lower() in query
            ):
                matched = claim
                break
        if not matched:
            result = {"found": False, "message": "No claim found for that drug and member."}
        else:
            result = {
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

    elif name == "get_formulary_alternatives":
        drug_class  = inputs["drug_class"].strip().lower()
        pharmacy_id = inputs["pharmacy_id"].strip()
        alts        = _DB["formulary_alternatives"].get(drug_class, [])
        inventory   = _DB["inventory"].get(pharmacy_id, {})
        enriched = []
        for alt in alts:
            stock = inventory.get(alt["drug"], {})
            enriched.append({
                **alt,
                "inventory_status": stock.get("status", "unknown"),
                "qty_on_hand":      stock.get("qty", 0),
            })
        result = {
            "drug_class":   drug_class,
            "pharmacy_id":  pharmacy_id,
            "alternatives": enriched,
        }

    elif name == "get_policy_status":
        eid    = inputs["emirates_id"].strip()
        member = _DB["members"].get(eid)
        if not member:
            result = {"found": False}
        else:
            result = {
                "found":         True,
                "policy_number": member["policy_number"],
                "insurer":       member["insurer"],
                "plan":          member["plan"],
                "status":        member["status"],
                "expiry_date":   member["expiry_date"],
            }

    else:
        result = {"error": f"Unknown tool: {name}"}

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    return result, elapsed_ms


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
3. If verification fails: "I'm unable to verify the identity on record." Do NOT reveal
   the stored name.

CLAIM & POLICY RULES
4. Use get_claim_status with the exact drug the caller mentions.
5. When a claim is "under_review" due to PA, explain:
   - Why PA is required (use pa_reason from tool result)
   - What the physician or pharmacy needs to submit
   - That review takes 24–48 hours after submission
6. When suggesting alternatives, include inventory status from get_formulary_alternatives
   so the pharmacy can act immediately.
7. If a policy is "expired", state this clearly and direct the caller to HR or the insurer.
   Do not process claims.
8. Use get_policy_status as a fast-path check when a claim is rejected.

VOICE BEHAVIOR
- This is a phone call. Keep responses concise — 2–4 sentences per turn.
- Do not use bullet points or headers in spoken replies.
- Be professional, warm, and efficient.
- Always use tools to fetch data. Never invent claim status, inventory, copays, or policy details.
"""


def build_system_prompt(tool_mode: str) -> str:
    if tool_mode == "json":
        return _SYSTEM_BASE + "\n" + _JSON_SHIM_ADDENDUM
    return _SYSTEM_BASE


# ═══════════════════════════════════════════════════════════════════════════════
# JSON-SHIM PARSER
# ═══════════════════════════════════════════════════════════════════════════════

_JSON_TOOL_RE = re.compile(
    r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)

# Catches Gemma-style ```tool_code\nfunc(arg="val")\n``` blocks
_GEMMA_TOOL_RE = re.compile(
    r'```(?:tool_code|python)?\s*\n(\w+)\(([^)]*)\)\s*\n```',
    re.DOTALL,
)

def _parse_gemma_tool_call(text: str) -> dict | None:
    """Parse Gemma's ```tool_code func(arg=val)``` format."""
    m = _GEMMA_TOOL_RE.search(text)
    if not m:
        return None
    fn_name = m.group(1)
    args_str = m.group(2).strip()
    # Parse keyword arguments: key="value" or key='value' or key=123
    args: dict = {}
    for kv in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\S+?)(?:,|$))', args_str):
        key = kv.group(1)
        val = kv.group(2) or kv.group(3) or kv.group(4)
        args[key] = val
    if fn_name and args:
        return {"name": fn_name, "arguments": args}
    return None

def parse_json_tool_call(text: str) -> dict | None:
    """Extract a JSON tool-call from raw model text. Returns {name, arguments} or None."""
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
# VLLM SERVER HELPERS  (run inside Modal containers)
# ═══════════════════════════════════════════════════════════════════════════════

def _start_vllm_server(cfg: ModelConfig, port: int) -> subprocess.Popen:
    """Launch the vLLM OpenAI-compatible HTTP server as a subprocess."""
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                cfg.hf_repo,
        "--download-dir",         MODEL_CACHE_PATH,
        "--dtype",                cfg.dtype,
        "--tensor-parallel-size", str(cfg.tensor_parallel),
        "--max-model-len",        str(cfg.max_model_len),
        "--port",                 str(port),
        "--no-enable-log-requests",
        "--trust-remote-code",
        "--hf-token", os.environ.get("HF_TOKEN", ""),
        "--attention-backend", "TRITON_ATTN",
        "--enforce-eager",
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
    env["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    print(f"[vLLM] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)


def _wait_for_vllm(port: int, timeout_s: int = 3600) -> None:
    """Block until vLLM health endpoint responds or timeout is reached."""
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
# CONVERSATION RUNNER  (model-agnostic, runs inside each Modal function)
# ═══════════════════════════════════════════════════════════════════════════════

def run_conversation(
    scenario_name: str,
    caller_turns: list[str],
    caller_type: str,
    cfg: ModelConfig,
    base_url: str,
) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="not-needed")

    sep  = "═" * 72
    sep2 = "─" * 72

    print(f"\n{sep}")
    print(f"  {scenario_name}")
    print(f"  Model       : {cfg.display_name}")
    print(f"  Tool mode   : {cfg.tool_mode}")
    print(f"  Caller type : {caller_type}")
    print(f"  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{sep}\n")

    messages: list[dict] = [{"role": "system", "content": build_system_prompt(cfg.tool_mode)}]
    tool_log: list[dict] = []
    turn_index = 0

    for caller_utterance in caller_turns:
        turn_index += 1
        print(f"{sep2}")
        print(f"  TURN {turn_index}")
        print(f"{sep2}")
        print(f"  CALLER : {caller_utterance}\n")

        messages.append({"role": "user", "content": caller_utterance})

        # ── Agentic tool-use loop ──────────────────────────────────────────────
        while True:

            # ── NATIVE tool-calling ──────────────────────────────────────────
            if cfg.tool_mode == "native":
                response = client.chat.completions.create(
                    model=cfg.hf_repo,
                    messages=messages,
                    tools=TOOLS_OPENAI,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=512,
                )
                msg = response.choices[0].message

                if msg.content and msg.content.strip():
                    print(f"  AGENT  : {msg.content.strip()}\n")

                if not msg.tool_calls:
                    messages.append({"role": "assistant", "content": msg.content or ""})
                    break

                # Preserve tool_calls in history as a plain dict (serialisable)
                messages.append({
                    "role":       "assistant",
                    "content":    msg.content or "",
                    "tool_calls": [
                        {
                            "id":   tc.id,
                            "type": "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    fn_name  = tc.function.name
                    fn_args  = json.loads(tc.function.arguments)
                    args_str = json.dumps(fn_args, ensure_ascii=False)

                    print(f"  [TOOL→] {fn_name}({args_str})")
                    result, elapsed_ms = execute_tool(fn_name, fn_args)
                    result_str = json.dumps(result, ensure_ascii=False)
                    print(f"  [←TOOL] {result_str}")
                    print(f"          latency: {elapsed_ms:.3f} ms\n")

                    tool_log.append({
                        "turn":       turn_index,
                        "tool":       fn_name,
                        "input":      fn_args,
                        "result":     result,
                        "latency_ms": round(elapsed_ms, 4),
                    })

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result_str,
                    })

            # ── JSON-SHIM tool-calling ───────────────────────────────────────
            else:
                response = client.chat.completions.create(
                    model=cfg.hf_repo,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=512,
                )
                raw_text = response.choices[0].message.content or ""
                tool_call = parse_json_tool_call(raw_text)

                if tool_call is None:
                    if raw_text.strip():
                        print(f"  AGENT  : {raw_text.strip()}\n")
                    messages.append({"role": "assistant", "content": raw_text})
                    break

                fn_name  = tool_call["name"]
                fn_args  = tool_call["arguments"]
                args_str = json.dumps(fn_args, ensure_ascii=False)

                print(f"  [TOOL→] {fn_name}({args_str})")
                result, elapsed_ms = execute_tool(fn_name, fn_args)
                result_str = json.dumps(result, ensure_ascii=False)
                print(f"  [←TOOL] {result_str}")
                print(f"          latency: {elapsed_ms:.3f} ms\n")

                tool_log.append({
                    "turn":       turn_index,
                    "tool":       fn_name,
                    "input":      fn_args,
                    "result":     result,
                    "latency_ms": round(elapsed_ms, 4),
                })

                # Inject result back as user message (JSON shim protocol)
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role":    "user",
                    "content": f"[TOOL RESULT for {fn_name}]: {result_str}",
                })

    # ── Tool timing summary ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  TOOL USE LOG — {cfg.display_name}")
    print(f"{sep}")
    print(f"  {'Turn':<6} {'Tool':<30} {'Latency':>10}")
    print(f"  {'─'*4:<6} {'─'*28:<30} {'─'*8:>10}")

    if tool_log:
        for e in tool_log:
            print(f"  {e['turn']:<6} {e['tool']:<30} {e['latency_ms']:>8.3f} ms")
        total_ms = sum(e["latency_ms"] for e in tool_log)
        avg_ms   = total_ms / len(tool_log)
        print(f"\n  Calls  : {len(tool_log)}")
        print(f"  Total  : {total_ms:.3f} ms")
        print(f"  Average: {avg_ms:.3f} ms")
    else:
        print("  (no tool calls made)")
    print(f"{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

_SCENARIOS: dict[int, dict] = {
    1: {
        "name": "Scenario 1 — Complex Pharmacy Call (Step Therapy & Alternatives)",
        "caller_type": "pharmacy_staff",
        "turns": [
            "Hi, I'm calling from Dubai Pharmacy branch 005. I'd like to check "
            "the status of a claim for a patient.",
            "Sure, I have his Emirates ID. It's 784-1996-7169603-3.",
            "Yes, it's Omar Ali.",
            "I am checking on a claim submitted for Zocor 40mg. Is it approved?",
            "Oh, okay. What are the covered alternatives that we can suggest to the physician?",
            "Perfect. I will contact the physician to adjust the prescription to Atorvastatin. "
            "Should I re-submit through E-Claim once it's updated?",
            "No, that's all. Thank you.",
        ],
    },
    2: {
        "name": "Scenario 2 — Standard Patient Call (Pending Prior Authorization)",
        "caller_type": "patient",
        "turns": [
            "Hello, I'm checking on the approval status for my prescription of Januvia.",
            "My Emirates ID is 784-2004-2137407-6. Date of birth is March 16th, 1988.",
            "Ahmed Khan.",
            "Yes, that's the one. Is it approved?",
            "How long will that take after my doctor sends it?",
            "No, I'll call my doctor now. Thanks.",
        ],
    },
    3: {
        "name": "Scenario 3 — Expired Policy (Hard-Stop Enforcement)",
        "caller_type": "patient",
        "turns": [
            "Hi, I just submitted a prescription for Plavix but the pharmacy said "
            "it wasn't going through. Can you check?",
            "Sure, Emirates ID is 784-1974-3341057-2. My birthday is May 5th, 1982.",
            "Fatima Al Mansoori.",
            "Oh really? I thought my company renewed it.",
            "Is there any way you can process it manually or give me a temporary override?",
            "Understood. I'll call HR right away. Thank you.",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# MODAL FUNCTIONS — one explicit global-scope function per model.
#
# Modal requires @app.function to decorate functions defined at global scope.
# Factories / closures don't work (hydration fails). Each function is identical
# in structure — only the ModelConfig it closes over differs.
#
# GPU string format: "RTX-PRO-6000" or "RTX-PRO-6000:N"  (Modal 1.x string API)
# ═══════════════════════════════════════════════════════════════════════════════

def _body(cfg: ModelConfig, scenario_ids: list[int]) -> None:
    """Shared implementation — start vLLM, run scenarios, shut down."""
    proc = _start_vllm_server(cfg, VLLM_PORT)
    try:
        _wait_for_vllm(VLLM_PORT, timeout_s=3600)
        base_url = f"http://localhost:{VLLM_PORT}/v1"
        for sid in scenario_ids:
            s = _SCENARIOS[sid]
            run_conversation(
                scenario_name=s["name"],
                caller_turns=s["turns"],
                caller_type=s["caller_type"],
                cfg=cfg,
                base_url=base_url,
            )
    finally:
        proc.terminate()
        proc.wait()


# ── 1× RTX-PRO-6000 ───────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    scaledown_window=300,
    min_containers=5,
    max_containers=5,
    allow_concurrent_inputs=1,
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_gpt120b_mxfp4():
    _body(MODEL_REGISTRY["gpt120b_mxfp4"])


@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_gemma4_26b(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gemma4_26b"], scenario_ids)


@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_gemma4_31b(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["gemma4_31b"], scenario_ids)


@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_qwen3_72b_fp8(scenario_ids: list[int]) -> None:
    _body(MODEL_REGISTRY["qwen3_72b_fp8"], scenario_ids)


@app.function(
    image=image,
    gpu="RTX-PRO-6000:2",
    scaledown_window=300,
    min_containers=5,
    max_containers=5,
    allow_concurrent_inputs=1,
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_qwen3_72b_bf16():
    _body(MODEL_REGISTRY["qwen3_72b_bf16"])


@app.function(
    image=image,
    gpu="RTX-PRO-6000:4",
    scaledown_window=300,
    min_containers=5,
    max_containers=5,
    allow_concurrent_inputs=1,
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_gpt120b_bf16():
    _body(MODEL_REGISTRY["gpt120b_bf16"])


# Registry maps model key → the Modal function object defined above
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
def main(
    model: str = "",
    scenario: int = 0,
) -> None:
    """
    Run scenarios on one or all models.

    Examples
    ────────
    modal run on_prem_oss.py
    modal run on_prem_oss.py --model gemma4_26b
    modal run on_prem_oss.py --model qwen3_72b_fp8 --scenario 2
    modal run on_prem_oss.py --scenario 3

    Available --model keys:
      gpt120b_mxfp4   GPT OSS 120B MXFP4 — 1× RTX PRO 6000
      gpt120b_bf16    GPT OSS 120B BF16  — 3× RTX PRO 6000
      gemma4_26b      Gemma 4 26B BF16   — 1× RTX PRO 6000
      gemma4_31b      Gemma 4 31B BF16   — 1× RTX PRO 6000
      qwen3_72b_fp8   Qwen3 72B FP8      — 1× RTX PRO 6000
      qwen3_72b_bf16  Qwen3 72B BF16     — 2× RTX PRO 6000
    """
    if model and model not in _RUNNERS:
        raise ValueError(
            f"Unknown model '{model}'. "
            f"Choose from: {', '.join(sorted(_RUNNERS))}"
        )

    target_models    = [model] if model else sorted(_RUNNERS)
    all_scenario_ids = sorted(_SCENARIOS)
    target_scenarios = [scenario] if scenario in all_scenario_ids else all_scenario_ids

    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  NGI Pharma OSS Voice AI — Demo Runner")
    print(f"  Models    : {target_models}")
    print(f"  Scenarios : {target_scenarios}")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{sep}\n")

    for model_key in target_models:
        cfg = MODEL_REGISTRY[model_key]

        print(f"\n{'─'*72}")
        print(f"  Launching  : {cfg.display_name}")
        print(f"  GPU spec   : RTX-PRO-6000 × {cfg.gpu_count}  ({cfg.vram_gb} GB est.)")
        print(f"  TP degree  : {cfg.tensor_parallel}")
        print(f"  dtype      : {cfg.dtype}")
        if cfg.quantization:
            print(f"  quant      : {cfg.quantization}")
        print(f"  tool mode  : {cfg.tool_mode}")
        print(f"{'─'*72}\n")

        _RUNNERS[model_key].remote(target_scenarios)