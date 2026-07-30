"""LangChain retriever backed by the SAMI RAG service.

``SamiRagRetriever`` calls the orchestrator RAG endpoint
(``POST /v1/rag/query``). The service performs retrieval *and* its own
RAGDefender pass, so the documents that come back are already context-poisoning
filtered. Each returned context document is surfaced as a LangChain
``Document`` whose metadata carries the request id, defense summary and policy
enforcement details for downstream auditing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field, PrivateAttr

from ._client import (
    build_orchestrator_api,
    build_rag_query_request,
    rag_call_kwargs,
)
from ._utils import rag_response_metadata


class SamiRagRetriever(BaseRetriever):
    """Retrieve guarded context documents from the SAMI RAG service.

    Example:
        .. code-block:: python

            from langchain_sami import SamiRagRetriever

            retriever = SamiRagRetriever(
                host="https://sami.example.com/rag-defender",
                top_k=5,
                tenant_id="acme",
            )
            docs = retriever.invoke("What is our SLA?")
    """

    host: Optional[str] = None
    """Base URL of the RAG service. Defaults to ``/rag-defender``."""

    access_token: Optional[str] = None
    """Optional bearer token forwarded as the ``Authorization`` header."""

    top_k: int = 10
    """Number of documents to request from the retriever (1-100)."""

    channel: Optional[str] = None
    """Channel identifier (``web``, ``api``, ``mobile`` ...)."""

    retriever_backend: Optional[str] = None
    """Optional retriever backend override (``weaviate`` ...)."""

    tenant_id: Optional[str] = None
    """Tenant id sent as the ``X-Tenant-Id`` header."""

    incident_id: Optional[str] = None
    """Existing firewall incident to attach this query to (``X-Incident-ID``)."""

    request_id: Optional[str] = None
    """Correlation id sent as the ``X-Request-ID`` header."""

    request_timeout: Optional[float] = Field(default=None, alias="timeout")
    """Per-request timeout in seconds."""

    client_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Extra ``Configuration`` keyword arguments."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    _orchestrator_api: Any = PrivateAttr(default=None)
    _last_response: Any = PrivateAttr(default=None)

    def _get_api(self) -> Any:
        if self._orchestrator_api is None:
            self._orchestrator_api = build_orchestrator_api(
                host=self.host,
                access_token=self.access_token,
                **self.client_kwargs,
            )
        return self._orchestrator_api

    def _run_query(self, query: str) -> Any:
        request = build_rag_query_request(
            query,
            top_k=self.top_k,
            channel=self.channel,
            retriever_backend=self.retriever_backend,
            incident_id=self.incident_id,
        )
        call_kwargs = rag_call_kwargs(
            access_token=self.access_token,
            request_id=self.request_id,
            incident_id=self.incident_id,
            tenant_id=self.tenant_id,
            request_timeout=self.request_timeout,
        )
        return self._get_api().rag_query(request, **call_kwargs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        response = self._run_query(query)
        self._last_response = response

        shared_metadata = rag_response_metadata(response)

        documents: List[Document] = []
        for index, content in enumerate(getattr(response, "context_docs", []) or []):
            metadata = dict(shared_metadata)
            metadata["doc_index"] = index
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    @property
    def last_response(self) -> Any:
        """The full :class:`RagQueryResponse` from the most recent call.

        Useful when you need the synthesized ``answer`` in addition to the
        retrieved context documents.
        """
        return self._last_response
