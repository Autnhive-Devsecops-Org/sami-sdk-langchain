"""Retrieve guarded context and get an end-to-end RAG answer.

Run:
    pip install -e "../[rag]"
    SAMI_RAG_HOST=https://sami.example.com/rag-defender python rag_retrieval.py
"""

import os

from langchain_sami import SamiRagChain, SamiRagRetriever


def main() -> None:
    host = os.environ.get("SAMI_RAG_HOST")
    token = os.environ.get("SAMI_TOKEN")

    backend = os.environ.get("SAMI_RETRIEVER_BACKEND")  # e.g. "weaviate"

    # Retriever: just the (defended) context documents.
    retriever = SamiRagRetriever(
        host=host,
        access_token=token,
        top_k=5,
        retriever_backend=backend,
        tenant_id=os.environ.get("SAMI_TENANT"),
    )
    docs = retriever.invoke("What is our uptime SLA?")
    print(f"Retrieved {len(docs)} documents")
    for doc in docs:
        print("-", doc.page_content[:120])
    if docs:
        print("Latency:", docs[0].metadata.get("latency_ms"))

    # Chain: the service-synthesized answer plus its supporting docs.
    chain = SamiRagChain(
        host=host, access_token=token, top_k=5, retriever_backend=backend
    )
    result = chain.invoke("What is our uptime SLA?")
    print("\nAnswer:", result.answer)
    print("Incident:", result.incident_id, "| log_type:", result.get("log_type"))
    print("Defense summary:", result.get("defense"))


if __name__ == "__main__":
    main()
