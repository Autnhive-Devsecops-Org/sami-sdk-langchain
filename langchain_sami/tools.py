"""LangChain tools exposing SAMI knowledge-base operations to agents.

These wrap the RAG ``SAMIApi`` management endpoints so an agent can ingest new
data and review quarantined documents, plus the RAGDefender defend endpoint for
ad-hoc context cleaning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ._client import build_rag_api_client
from ._utils import to_serializable


def _rag_apis(
    host: Optional[str],
    access_token: Optional[str],
    client_kwargs: Optional[Dict[str, Any]],
) -> Any:
    import sami_rag_client

    api_client = build_rag_api_client(
        host=host, access_token=access_token, **(client_kwargs or {})
    )
    return sami_rag_client


# --------------------------------------------------------------------------- #
# Input schemas
# --------------------------------------------------------------------------- #
class IngestSyncInput(BaseModel):
    """Synchronously ingest documents into the RAG knowledge base."""

    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier")
    app_id: Optional[str] = Field(default=None, description="Application identifier")
    bucket: Optional[str] = Field(default=None, description="Source bucket name")
    data_source_id: Optional[str] = Field(
        default=None, description="Data source identifier"
    )
    files: Optional[List[str]] = Field(
        default=None, description="Explicit list of files/paths to ingest"
    )
    store_quarantine: bool = Field(
        default=True, description="Whether flagged docs are quarantined for review"
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


class DefendInput(BaseModel):
    """Run RAGDefender over an explicit query + document list."""

    query: str = Field(description="The user query the documents will answer")
    documents: List[str] = Field(description="Retrieved document texts to filter")
    mode: str = Field(default="multihop", description="Defense mode")
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier")


# --------------------------------------------------------------------------- #
# Tool factory
# --------------------------------------------------------------------------- #
def make_sami_rag_tools(
    host: Optional[str] = None,
    access_token: Optional[str] = None,
    *,
    tenant_id: Optional[str] = None,
    client_kwargs: Optional[Dict[str, Any]] = None,
) -> List[StructuredTool]:
    """Build LangChain tools bound to a RAG service configuration.

    :param host: RAG service base URL.
    :param access_token: Optional bearer token.
    :param tenant_id: Default tenant id applied when a call omits one.
    :param client_kwargs: Extra ``Configuration`` keyword arguments.
    :returns: ``[ingest_sync, approve_quarantine, reject_quarantine, defend]``.
    """

    def _ingest_sync(**kwargs: Any) -> Dict[str, Any]:
        sami_rag_client = _rag_apis(host, access_token, client_kwargs)
        api = sami_rag_client.SAMIApi(
            build_rag_api_client(host, access_token, **(client_kwargs or {}))
        )
        data = IngestSyncInput(**kwargs)
        request = sami_rag_client.IngestSyncRequest(
            tenant_id=data.tenant_id or tenant_id,
            app_id=data.app_id,
            bucket=data.bucket,
            data_source_id=data.data_source_id,
            files=data.files,
            store_quarantine=data.store_quarantine,
        )
        return to_serializable(api.ingest_sync(ingest_sync_request=request))

    def _approve_quarantine(**kwargs: Any) -> Dict[str, Any]:
        sami_rag_client = _rag_apis(host, access_token, client_kwargs)
        api = sami_rag_client.SAMIApi(
            build_rag_api_client(host, access_token, **(client_kwargs or {}))
        )
        data = QuarantineReviewInput(**kwargs)
        review = sami_rag_client.QuarantineReviewRequest(
            reason=data.reason,
            incident_id=data.incident_id,
            data_source_id=data.data_source_id,
            bucket=data.bucket,
        )
        return to_serializable(
            api.approve_quarantine_doc(data.doc_id, quarantine_review_request=review)
        )

    def _reject_quarantine(**kwargs: Any) -> Dict[str, Any]:
        sami_rag_client = _rag_apis(host, access_token, client_kwargs)
        api = sami_rag_client.SAMIApi(
            build_rag_api_client(host, access_token, **(client_kwargs or {}))
        )
        data = QuarantineReviewInput(**kwargs)
        review = sami_rag_client.QuarantineReviewRequest(
            reason=data.reason,
            incident_id=data.incident_id,
            data_source_id=data.data_source_id,
            bucket=data.bucket,
        )
        return to_serializable(
            api.reject_quarantine_doc(data.doc_id, quarantine_review_request=review)
        )

    def _defend(**kwargs: Any) -> Dict[str, Any]:
        sami_rag_client = _rag_apis(host, access_token, client_kwargs)
        api = sami_rag_client.DEFENDERApi(
            build_rag_api_client(host, access_token, **(client_kwargs or {}))
        )
        data = DefendInput(**kwargs)
        request = sami_rag_client.DefendRequestModel(
            query=data.query,
            documents=data.documents,
            mode=data.mode,
            tenant_id=data.tenant_id or tenant_id,
        )
        return to_serializable(api.defend_v1_defend_post(request))

    return [
        StructuredTool.from_function(
            func=_ingest_sync,
            name="sami_ingest_sync",
            description=(
                "Synchronously ingest documents into the SAMI RAG knowledge base. "
                "Returns counts of accepted / quarantined / rejected documents."
            ),
            args_schema=IngestSyncInput,
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
        StructuredTool.from_function(
            func=_defend,
            name="sami_defend_documents",
            description=(
                "Run the SAMI RAGDefender over a query and a list of document "
                "texts; returns the cleaned/kept documents and per-doc scores."
            ),
            args_schema=DefendInput,
        ),
    ]
