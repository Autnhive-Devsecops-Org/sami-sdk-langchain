"""LangChain tools exposing SAMI knowledge-base operations to agents.

These wrap the RAG ``SAMIApi`` management endpoints so an agent can ingest new
data and review quarantined documents:

* ``POST /v1/ingest``                     -> ``sami_ingest_file_url``
* ``POST /v1/quarantine/{doc_id}/approve`` -> ``sami_approve_quarantine``
* ``POST /v1/quarantine/{doc_id}/reject``  -> ``sami_reject_quarantine``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ._client import _import_rag, build_sami_api, rag_call_kwargs
from ._utils import to_serializable


# --------------------------------------------------------------------------- #
# Input schemas
# --------------------------------------------------------------------------- #
class IngestFileUrlInput(BaseModel):
    """Ingest a single document into the RAG knowledge base from its URL."""

    file_url: str = Field(
        description="URL of the file to ingest (signed URL or publicly reachable link)"
    )
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier")
    app_id: Optional[str] = Field(default=None, description="Application identifier")
    doc_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional public document identifier. When omitted a stable id is "
            "derived from the file URL."
        ),
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Arbitrary metadata stored alongside the document"
    )
    store_quarantine: bool = Field(
        default=True, description="Whether flagged docs are quarantined for review"
    )
    retriever_backend: Optional[str] = Field(
        default=None, description="Retriever backend override (e.g. 'weaviate')"
    )


class QuarantineReviewInput(BaseModel):
    """Approve or reject a quarantined document."""

    doc_id: str = Field(description="Identifier of the quarantined document")
    reason: Optional[str] = Field(
        default=None, description="Reviewer reason for the decision"
    )
    incident_id: Optional[str] = Field(default=None, description="Related incident id")
    data_source_id: Optional[str] = Field(default=None, description="Data source id")
    bucket: Optional[str] = Field(default=None, description="Source bucket name")
    retriever_backend: Optional[str] = Field(
        default=None, description="Retriever backend override (e.g. 'weaviate')"
    )


# --------------------------------------------------------------------------- #
# Tool factory
# --------------------------------------------------------------------------- #
def make_sami_rag_tools(
    host: Optional[str] = None,
    access_token: Optional[str] = None,
    *,
    tenant_id: Optional[str] = None,
    request_timeout: Optional[float] = None,
    client_kwargs: Optional[Dict[str, Any]] = None,
) -> List[StructuredTool]:
    """Build LangChain tools bound to a RAG service configuration.

    :param host: RAG service base URL.
    :param access_token: Optional bearer token.
    :param tenant_id: Default tenant id applied when a call omits one; also sent
        as the ``X-Tenant-Id`` header.
    :param request_timeout: Per-request timeout in seconds.
    :param client_kwargs: Extra ``Configuration`` keyword arguments.
    :returns: ``[ingest_file_url, approve_quarantine, reject_quarantine]``.
    """
    # The SAMIApi endpoints take no X-Request-ID / X-Incident-ID header params.
    call_kwargs = rag_call_kwargs(
        access_token=access_token,
        tenant_id=tenant_id,
        request_timeout=request_timeout,
        header_params=False,
    )

    def _api() -> Any:
        return build_sami_api(host, access_token, **(client_kwargs or {}))

    def _review_request(data: QuarantineReviewInput) -> Any:
        sami_rag_client = _import_rag()
        return sami_rag_client.QuarantineReviewRequest(
            reason=data.reason,
            incident_id=data.incident_id,
            data_source_id=data.data_source_id,
            bucket=data.bucket,
            retriever_backend=data.retriever_backend,
        )

    def _ingest_file_url(**kwargs: Any) -> Dict[str, Any]:
        sami_rag_client = _import_rag()
        data = IngestFileUrlInput(**kwargs)
        request = sami_rag_client.FileUrlIngestRequest(
            tenant_id=data.tenant_id or tenant_id,
            app_id=data.app_id,
            doc_id=data.doc_id,
            file_url=data.file_url,
            metadata=data.metadata,
            store_quarantine=data.store_quarantine,
            retriever_backend=data.retriever_backend,
        )
        return to_serializable(
            _api().ingest_commit(file_url_ingest_request=request, **call_kwargs)
        )

    def _approve_quarantine(**kwargs: Any) -> Dict[str, Any]:
        data = QuarantineReviewInput(**kwargs)
        return to_serializable(
            _api().approve_quarantine_doc(
                data.doc_id,
                quarantine_review_request=_review_request(data),
                **call_kwargs,
            )
        )

    def _reject_quarantine(**kwargs: Any) -> Dict[str, Any]:
        data = QuarantineReviewInput(**kwargs)
        return to_serializable(
            _api().reject_quarantine_doc(
                data.doc_id,
                quarantine_review_request=_review_request(data),
                **call_kwargs,
            )
        )

    return [
        StructuredTool.from_function(
            func=_ingest_file_url,
            name="sami_ingest_file_url",
            description=(
                "Ingest a document into the SAMI RAG knowledge base from its file "
                "URL. Returns accepted / quarantined / rejected counts and the "
                "resulting document ids."
            ),
            args_schema=IngestFileUrlInput,
        ),
        StructuredTool.from_function(
            func=_approve_quarantine,
            name="sami_approve_quarantine",
            description=(
                "Approve a quarantined document by id so it becomes searchable."
            ),
            args_schema=QuarantineReviewInput,
        ),
        StructuredTool.from_function(
            func=_reject_quarantine,
            name="sami_reject_quarantine",
            description="Reject a quarantined document by id so it is discarded.",
            args_schema=QuarantineReviewInput,
        ),
    ]
