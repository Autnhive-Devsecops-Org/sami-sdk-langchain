# langchain-sami

LangChain integrations for the **SAMI AI Firewall** and **SAMI RAG** services.

This package is a thin, idiomatic LangChain layer on top of the two generated
OpenAPI SDKs:

| SDK | Branch | Service | Wrapped by |
| --- | --- | --- | --- |
| `sami_firewall_client` | `master` | AI Firewall (`/ai-firewall/firewall/...`) | `ChatSamiFirewall` |
| `sami_rag_client` | `SDK_RAG_TEMP_ORPHAN` | RAG (`/rag-defender/...`) | `SamiRagRetriever`, `SamiRagChain`, tools |

## Installation

Install this package together with the SDK(s) you need. The generated clients
live on two different branches of the SDK repo:

```sh
# 1. The generated clients (pick what you need)
pip install "git+https://github.com/Autnhive-Devsecops-Org/sami-sdk-python.git"                       # firewall (master)
pip install "git+https://github.com/Autnhive-Devsecops-Org/sami-sdk-python.git@SDK_RAG_TEMP_ORPHAN"   # rag

# 2. This LangChain layer
pip install -e ./sami-langchain
```

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
| `make_sami_rag_tools(...)` (`StructuredTool[]`) | ingest / quarantine | Agent tools to manage the knowledge base. |

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
    top_k=5,
    tenant_id="acme",
)
docs = retriever.invoke("What is our SLA?")
```

### 3. End-to-end RAG answer

```python
from langchain_sami import SamiRagChain

chain = SamiRagChain(host="https://sami.example.com/rag-defender")
result = chain.invoke("How do I rotate my API key?")
print(result.answer)
```

### 4. Agent tools

```python
from langchain_sami import make_sami_rag_tools

tools = make_sami_rag_tools(
    host="https://sami.example.com/rag-defender", tenant_id="acme"
)
# -> [sami_ingest_sync, sami_approve_quarantine, sami_reject_quarantine]
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
