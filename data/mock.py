import random
import uuid
from datetime import datetime, timedelta
import json

PBMS = ["NAS", "Daman", "AXA Gulf", "ADNIC", "Cigna ME"]
PLANS = ["Thiqa", "Basic", "Enhanced", "Gold"]
DRUGS = [
    ("Atorvastatin 20mg", "Statin"),
    ("Metformin 500mg", "Biguanide"),
    ("Insulin Glargine", "Insulin"),
    ("Clopidogrel 75mg", "Antiplatelet"),
    ("Amoxicillin 500mg", "Antibiotic")
]
REJECTIONS = [
    ("79", "Prior Authorization Required"),
    ("70", "Product/Service Not Covered"),
    ("75", "PA Required – Step Therapy"),
    ("76", "Plan Limit Exceeded"),
    ("M1", "Missing Information")
]
ACCENTS = ["Gulf Arabic", "Indian English", "Pakistani English", "Filipino English"]

def random_time():
    base = datetime(2025, 1, 1)
    return (base + timedelta(days=random.randint(0, 30))).isoformat()

records = []

for i in range(200):
    drug, drug_class = random.choice(DRUGS)
    ddc, reason = random.choice(REJECTIONS)

    metadata = {
        "pharmacy_id": f"DXB-PH-{random.randint(1,30):03}",
        "pbm_name": random.choice(PBMS),
        "insurance_plan": random.choice(PLANS),
        "patient_age": random.randint(18, 85),
        "drug_name": drug,
        "drug_class": drug_class,
        "ndc_code": f"{random.randint(10000,99999)}-{random.randint(1000,9999)}-01",
        "ddc_code": ddc,
        "rejection_reason": reason,
        "call_outcome": random.choice(["Resolved", "Escalated", "Callback Required"]),
        "resolution_action": random.choice([
            "PA fax requested",
            "Clinical notes submitted",
            "Physician approval pending",
            "Coverage confirmed"
        ]),
        "call_duration_sec": random.randint(180, 900),
        "agent_type": "AI",
        "accent_detected": random.choice(ACCENTS),
        "timestamp": random_time(),
        "compliance_flag": False
    }

    embedding_text = (
        f"PBM rejection call. Insurance: {metadata['pbm_name']}, "
        f"Plan: {metadata['insurance_plan']}. "
        f"Drug: {drug} ({drug_class}), NDC {metadata['ndc_code']}. "
        f"Rejection reason: {reason}. "
        f"DDC code: {ddc}. "
        f"Pharmacy action: {metadata['resolution_action']}. "
        f"Call outcome: {metadata['call_outcome']}. "
        f"Accent: {metadata['accent_detected']}."
    )

    records.append({
        "id": f"pbm_call_{i:04}",
        "text_for_embedding": embedding_text,
        "metadata": metadata
    })

    with open("data/pharma.json", "w") as f:
        json.dump(records, f, indent=4)