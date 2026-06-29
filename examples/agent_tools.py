"""Expose SAMI knowledge-base operations as LangChain agent tools.

Run:
    pip install -e "../[rag]"
    SAMI_RAG_HOST=https://sami.example.com/rag-defender python agent_tools.py
"""

import os

from langchain_sami import make_sami_rag_tools


def main() -> None:
    tools = make_sami_rag_tools(
        host=os.environ.get("SAMI_RAG_HOST"),
        access_token=os.environ.get("SAMI_TOKEN"),
        tenant_id=os.environ.get("SAMI_TENANT", "acme"),
    )

    for tool in tools:
        print(f"{tool.name}: {tool.description}")

    # Direct invocation (an agent would call these via tool-calling):
    ingest = next(t for t in tools if t.name == "sami_ingest_sync")
    result = ingest.invoke({"data_source_id": "s3-handbook", "store_quarantine": True})
    print("\nIngest result:", result)


if __name__ == "__main__":
    main()
