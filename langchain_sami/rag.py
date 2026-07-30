"""A ready-made RAG runnable backed by the SAMI orchestrator.

Unlike :class:`SamiRagRetriever`, which returns only the context documents, the
``SamiRagChain`` returns the service-synthesized answer together with its
supporting context and defense metadata. The RAG service already runs retrieval,
the RAGDefender filter and the LLM call server-side, so this is a single network
round-trip.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from pydantic import ConfigDict, Field, PrivateAttr

from ._client import (
    build_orchestrator_api,
    build_rag_query_request,
    rag_call_kwargs,
)
from ._utils import rag_response_metadata, to_serializable


class SamiRagAnswer(Dict[str, Any]):
    """Dict subclass returned by :class:`SamiRagChain` with handy accessors."""

    @property
    def answer(self) -> str:
        return self.get("answer", "")

    @property
    def documents(self) -> List[Document]:
        return self.get("documents", [])

    @property
    def incident_id(self) -> Optional[str]:
        """Firewall incident this query was recorded against, if any."""
        return self.get("incident_id")


class SamiRagChain(RunnableSerializable[Union[str, Dict[str, Any]], SamiRagAnswer]):
    """End-to-end RAG runnable over the SAMI orchestrator.

    Invoke it with either a bare query string or a ``{"query": ...}`` dict::

        from langchain_sami import SamiRagChain

        chain = SamiRagChain(host="https://sami.example.com/rag-defender")
        result = chain.invoke("How do I rotate my API key?")
        print(result.answer)
        for doc in result.documents:
            print(doc.page_content)
    """

    host: Optional[str] = None
    access_token: Optional[str] = None
    top_k: int = 10
    channel: Optional[str] = None
    retriever_backend: Optional[str] = None
    tenant_id: Optional[str] = None
    """Tenant id sent as the ``X-Tenant-Id`` header."""

    incident_id: Optional[str] = None
    """Existing firewall incident to attach this query to (``X-Incident-ID``)."""

    request_id: Optional[str] = None
    """Correlation id sent as the ``X-Request-ID`` header."""

    request_timeout: Optional[float] = Field(default=None, alias="timeout")
    client_kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    _orchestrator_api: Any = PrivateAttr(default=None)

    def _get_api(self) -> Any:
        if self._orchestrator_api is None:
            self._orchestrator_api = build_orchestrator_api(
                host=self.host,
                access_token=self.access_token,
                **self.client_kwargs,
            )
        return self._orchestrator_api

    @staticmethod
    def _coerce_query(value: Union[str, Dict[str, Any]]) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("query", "question", "input"):
                if key in value:
                    return str(value[key])
        raise ValueError(
            "SamiRagChain input must be a query string or a dict containing a "
            "'query' key."
        )

    def invoke(
        self,
        input: Union[str, Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> SamiRagAnswer:
        query = self._coerce_query(input)
        overrides = input if isinstance(input, dict) else {}
        incident_id = overrides.get("incident_id", self.incident_id)
        request_id = overrides.get("request_id", self.request_id)

        request = build_rag_query_request(
            query,
            top_k=self.top_k,
            channel=self.channel,
            retriever_backend=self.retriever_backend,
            incident_id=incident_id,
        )
        call_kwargs = rag_call_kwargs(
            access_token=self.access_token,
            request_id=request_id,
            incident_id=incident_id,
            tenant_id=self.tenant_id,
            request_timeout=self.request_timeout,
        )

        response = self._get_api().rag_query(request, **call_kwargs)

        metadata = rag_response_metadata(response, include_defense=False)
        documents = [
            Document(page_content=content, metadata={**metadata, "doc_index": i})
            for i, content in enumerate(getattr(response, "context_docs", []) or [])
        ]

        return SamiRagAnswer(
            query=query,
            answer=getattr(response, "answer", ""),
            documents=documents,
            defense=to_serializable(getattr(response, "defense", None)),
            policy_enforcement=to_serializable(
                getattr(response, "policy_enforcement", None)
            ),
            **metadata,
        )
