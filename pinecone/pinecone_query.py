from pinecone import Pinecone
import os
from decouple import config

PINECONE_API_KEY = config("PINECONE_API_KEY")
PINECONE_HOST = config("PINECONE_HOST")
PINECONE_NAMESPACE = config("PINECONE_NAMESPACE")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

def semantic_search(query_text: str, top_k: int = 3):
    """
    Perform semantic search using text input.
    Returns the top_k matching records.
    """

    response = index.search(
        namespace=PINECONE_NAMESPACE,
        query={
            "inputs": {"text": query_text},
            "top_k": top_k
        },
        fields=["text", *[]],  # you can include other fields if needed
    )

    return response

if __name__ == "__main__":
    query = "Check prior authorization rejection issues for insulin"
    results = semantic_search(query, top_k=3)
    print(results['result']['hits'])