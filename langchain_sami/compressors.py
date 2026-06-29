"""RAGDefender as a LangChain document compressor.

The defend endpoint (``POST /v1/defend``) takes a query plus a list of retrieved
documents and returns the subset that survives context-poisoning / prompt
injection filtering. That maps exactly onto LangChain's
:class:`~langchain_core.documents.compressor.BaseDocumentCompressor`
abstraction, so you can drop it into a ``ContextualCompressionRetriever`` to
guard *any* existing retriever:

    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_sami import SamiRagDefenderCompressor

    guarded = ContextualCompressionRetriever(
        base_compressor=SamiRagDefenderCompressor(host=..., tenant_id="acme"),
        base_retriever=my_vectorstore.as_retriever(),
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import ConfigDict, Field, PrivateAttr

from ._client import build_rag_api_client
from ._utils import to_serializable


class SamiRagDefenderCompressor(BaseDocumentCompressor):
    """Filter retrieved documents through the SAMI RAGDefender."""

    host: Optional[str] = None
    access_token: Optional[str] = None
    mode: str = "multihop"
    """Defense mode (``multihop`` by default)."""
    top_k: Optional[int] = None
    """Optional cap on the number of documents returned."""
    tenant_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    """Additional metadata forwarded to the defend endpoint."""
    request_timeout: Optional[float] = Field(default=None, alias="timeout")
    client_kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    _defender_api: Any = PrivateAttr(default=None)

    def _get_api(self) -> Any:
        if self._defender_api is None:
            import sami_rag_client

            api_client = build_rag_api_client(
                host=self.host,
                access_token=self.access_token,
                **self.client_kwargs,
            )
            self._defender_api = sami_rag_client.DEFENDERApi(api_client)
        return self._defender_api

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        documents = list(documents)
        if not documents:
            return documents

        import sami_rag_client

        request = sami_rag_client.DefendRequestModel(
            query=query,
            documents=[doc.page_content for doc in documents],
            mode=self.mode,
            top_k=self.top_k,
            tenant_id=self.tenant_id,
            metadata=self.metadata,
        )
        call_kwargs: Dict[str, Any] = {}
        if self.request_timeout is not None:
            call_kwargs["_request_timeout"] = self.request_timeout
        if self.tenant_id is not None:
            call_kwargs["x_tenant_id"] = self.tenant_id

        response = self._get_api().defend_v1_defend_post(request, **call_kwargs)

        kept_indices = list(getattr(response, "kept_indices", []) or [])
        scores = getattr(response, "document_scores", []) or []
        stats = to_serializable(getattr(response, "stats", None))

        kept_docs: List[Document] = []
        for position, original_index in enumerate(kept_indices):
            if original_index >= len(documents):
                continue
            source = documents[original_index]
            metadata = dict(source.metadata)
            metadata["sami_defense"] = {
                "kept": True,
                "original_index": original_index,
                "request_id": getattr(response, "request_id", None),
                "stats": stats,
            }
            if position < len(scores):
                metadata["sami_defense"]["score"] = to_serializable(scores[position])
            kept_docs.append(
                Document(page_content=source.page_content, metadata=metadata)
            )

        # If the service did not report indices, fall back to the cleaned text.
        if not kept_indices:
            for content in getattr(response, "clean_documents", []) or []:
                kept_docs.append(Document(page_content=content, metadata={}))

        return kept_docs
