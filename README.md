# langchain-sami

LangChain integrations for the **SAMI AI Firewall** and **SAMI RAG** services.

This package is a thin, idiomatic LangChain layer on top of the two generated
OpenAPI SDKs:

| SDK | Repo | Service | Wrapped by |
| --- | --- | --- | --- |
| `sami_firewall_client` | `sami-sdk-python` | AI Firewall (`/ai-firewall/firewall/...`) | `ChatSamiFirewall` |
| `sami_rag_client` | `sami-rag-sdk-python` | RAG (`/rag-defender/...`) | `SamiRagRetriever`, `SamiRagChain`, tools |

## Installation

Install this package together with the SDK(s) you need. The generated clients
live in two separate repos:

```sh
# 1. The generated clients (pick what you need)
pip install "git+https://github.com/Autnhive-Devsecops-Org/sami-sdk-python.git"       # firewall
pip install "git+https://github.com/Autnhive-Devsecops-Org/sami-rag-sdk-python.git"   # rag

# 2. This LangChain layer
pip install -e ./sami-langchain
```

> The RAG client used to be an orphan branch (`SDK_RAG_TEMP_ORPHAN`) of
> `sami-sdk-python`. It now has its own repo, and the ingest endpoint changed
> from a synchronous multi-file call to `POST /v1/ingest` with a single file URL
> — see [Migration](#migration-notes) below.

Or let the extras pull the clients for you:

```sh
pip install -e "./sami-langchain[all]"      # both clients
pip install -e "./sami-langchain[firewall]" # firewall only
pip install -e "./sami-langchain[rag]"      # rag only
```

## What each component maps to

| Component | Endpoint | Purpose |
| --- | --- | --- |
| `ChatSamiFirewall` (`BaseChatModel`) | `POST /ai-firewall/firewall/v1/prompt/text` | A guarded chat model; prompts are sanitised by the firewall before reaching the LLM. |
| `SamiRagRetriever` (`BaseRetriever`) | `POST /v1/rag/query` | Returns RAGDefender-filtered context documents as LangChain `Document`s. |
| `SamiRagChain` (`Runnable`) | `POST /v1/rag/query` | One-shot RAG: returns the synthesized `answer` plus supporting docs. |
| `make_sami_rag_tools(...)` (`StructuredTool[]`) | `POST /v1/ingest`, `POST /v1/quarantine/{doc_id}/{approve,reject}` | Agent tools to manage the knowledge base. |

## Quick start

### 1. Firewall-guarded chat model

```python
from langchain_sami import ChatSamiFirewall

llm = ChatSamiFirewall(
    host="https://sami.example.com",
    access_token="<HTTPBearer token>",
    ai_provider="openai",
    ai_key="sk-...",
)
print(llm.invoke("Summarise our refund policy.").content)
```

### 2. RAG retriever

```python
from langchain_sami import SamiRagRetriever

retriever = SamiRagRetriever(
    host="https://sami.example.com/rag-defender",
    access_token="sk_llm-...",
    top_k=5,
    tenant_id="acme",
    retriever_backend="weaviate",
)
docs = retriever.invoke("What is our SLA?")
docs[0].metadata  # request_id, incident_id, log_type, latency_ms, defense, ...
```

### 3. End-to-end RAG answer

```python
from langchain_sami import SamiRagChain

chain = SamiRagChain(host="https://sami.example.com/rag-defender")
result = chain.invoke("How do I rotate my API key?")
print(result.answer, result.incident_id)

# Attach the query to an existing firewall incident (per call):
result = chain.invoke({"query": "...", "incident_id": "inc_123"})
```

### 4. Agent tools

```python
from langchain_sami import make_sami_rag_tools

tools = make_sami_rag_tools(
    host="https://sami.example.com/rag-defender", tenant_id="acme"
)
# -> [sami_ingest_file_url, sami_approve_quarantine, sami_reject_quarantine]

tools[0].invoke({"file_url": "https://.../handbook.pdf", "store_quarantine": True})
```

See [`examples/`](./examples) for runnable scripts.

## Design notes

* The SDKs are imported **lazily**, so you only need the client(s) for the
  components you actually use.
* `host` / `access_token` / extra `Configuration` kwargs are accepted by every
  component and forwarded to the generated `Configuration`.
* The firewall `adapter_chat` endpoint is typed `Dict[str, object]`; the chat
  model extracts assistant text defensively (OpenAI `choices`, flat
  `content`/`answer`/`response` keys) and always stashes the raw payload in
  `response_metadata["raw"]`.
* The generated RAG `Configuration.auth_settings()` is empty, so the token is
  passed per call as the `authorization` argument (`"Bearer <token>"`); see
  `_client.bearer_header`.
* `_client.py` is the only module that names generated API classes, request
  models and per-call arguments, so a regenerated SDK is absorbed in one place.

## Migration notes

Changes required by the current `sami_rag_client` (0.2.2, `sami-rag-sdk-python`):

| Before | Now |
| --- | --- |
| `pip install ...sami-sdk-python.git@SDK_RAG_TEMP_ORPHAN` | `pip install ...sami-rag-sdk-python.git` |
| `sami_ingest_sync` tool (`SAMIApi.ingest_sync` + `IngestSyncRequest`) — **removed from the SDK** | `sami_ingest_file_url` tool (`SAMIApi.ingest_commit` + `FileUrlIngestRequest`, one `file_url` per call) |
| Ingest args `bucket` / `data_source_id` / `files` | `file_url`, plus optional `doc_id`, `metadata`, `retriever_backend` |
| `tenant_id` on the retriever was documented but never sent | Sent as the `X-Tenant-Id` header (also available on `SamiRagChain`) |
| — | `incident_id` / `request_id` on the retriever and chain (`X-Incident-ID`, `X-Request-ID`, `RagQueryRequest.incident_id`) |
| Metadata: `request_id`, `tenant_id`, `app_id` | Also `incident_id`, `log_type`, `latency_ms` |
