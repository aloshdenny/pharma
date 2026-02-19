from pinecone import Pinecone
import json
from decouple import config

PINECONE_API_KEY = config("PINECONE_API_KEY")
PINECONE_HOST = config("PINECONE_HOST")
PINECONE_NAMESPACE = config("PINECONE_NAMESPACE")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

def sanitize_metadata(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [str(v) for v in value]
    return json.dumps(value, ensure_ascii=False)

def batch_upsert(records, batch_size=10):
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        pinecone_records = []

        for record in batch:
            record_id = record["id"]

            payload = {k: v for k, v in record.items() if k != "id"}

            sanitized_metadata = {
                k: sanitize_metadata(v)
                for k, v in payload.items()
                if sanitize_metadata(v) is not None
            }

            pinecone_records.append({
                "_id": record_id,
                "text": json.dumps(payload, ensure_ascii=False),
                **sanitized_metadata
            })

        index.upsert_records(
            namespace=PINECONE_NAMESPACE,
            records=pinecone_records
        )

        print(f"Upserted batch {i // batch_size + 1}")


# -------- RUN --------
with open("data/db.json", "r") as f:
    records = json.load(f)

batch_upsert(records)