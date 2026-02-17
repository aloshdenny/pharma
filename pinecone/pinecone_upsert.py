from pinecone import Pinecone
import json
from decouple import config
from sentence_transformers import SentenceTransformer

PINECONE_API_KEY = config("PINECONE_API_KEY")
PINECONE_HOST = config("PINECONE_HOST")
PINECONE_NAMESPACE = config("PINECONE_NAMESPACE")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

with open("pharma.json", "r") as f:
    records = json.load(f)

def batch_upsert(records, batch_size=50):
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]

        pinecone_records = []

        for r in batch:
            # build record WITHOUT changing source data
            record = {
                "_id": r["id"],
                "text": r["text_for_embedding"],  # <-- REQUIRED FIELD NAME
            }

            # flatten metadata into top-level fields
            for k, v in r["metadata"].items():
                record[k] = v

            pinecone_records.append(record)

        index.upsert_records(
            namespace=PINECONE_NAMESPACE,
            records=pinecone_records
        )

        print(f"Upserted batch {i // batch_size + 1}")

batch_upsert(records)