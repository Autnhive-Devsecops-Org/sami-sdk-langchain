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

    # Retriever: just the (defended) context documents.
    retriever = SamiRagRetriever(host=host, access_token=token, top_k=5)
    docs = retriever.invoke("What is our uptime SLA?")
    print(f"Retrieved {len(docs)} documents")
    for doc in docs:
        print("-", doc.page_content[:120])

    # Chain: the service-synthesized answer plus its supporting docs.
    chain = SamiRagChain(host=host, access_token=token, top_k=5)
    result = chain.invoke("What is our uptime SLA?")
    print("\nAnswer:", result.answer)
    print("Defense summary:", result.get("defense"))


if __name__ == "__main__":
    main()
