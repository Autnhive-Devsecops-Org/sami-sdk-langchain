"""Helpers for constructing the underlying SAMI OpenAPI clients.

The two generated SDKs live in separate repositories and are installed
separately:

    # RAG client  (sami-rag-sdk-python, package sami_rag_client)
    pip install "git+https://github.com/Autnhive-Devsecops-Org/sami-rag-sdk-python.git"

    # Firewall client  (sami-sdk-python, package sami_firewall_client)
    pip install "git+https://github.com/Autnhive-Devsecops-Org/sami-sdk-python.git"

This module imports them lazily so that, for example, a project that only uses
the firewall chat model does not need the RAG client installed and vice versa.
It is also the single place that knows the generated API class names and the
per-call argument conventions, so a regenerated SDK only has to be tracked here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

_FIREWALL_INSTALL = (
    'pip install "git+https://github.com/Autnhive-Devsecops-Org/'
    'sami-sdk-python.git"'
)
_RAG_INSTALL = (
    'pip install "git+https://github.com/Autnhive-Devsecops-Org/'
    'sami-rag-sdk-python.git"'
)


def _import_firewall() -> Any:
    try:
        import sami_firewall_client  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "The 'sami_firewall_client' package is required for firewall "
            f"features. Install it with:\n    {_FIREWALL_INSTALL}"
        ) from exc
    return sami_firewall_client


def _import_rag() -> Any:
    try:
        import sami_rag_client  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "The 'sami_rag_client' package is required for RAG features. "
            f"Install it with:\n    {_RAG_INSTALL}"
        ) from exc
    return sami_rag_client


def bearer_header(access_token: Optional[str]) -> Optional[str]:
    """Return the value to pass as the RAG SDK's per-call ``authorization`` arg.

    Unlike the firewall client, the generated RAG client's
    ``Configuration.auth_settings()`` is empty, so setting ``access_token`` on
    the configuration never injects an ``Authorization`` header. Every RAG API
    method instead accepts an ``authorization`` keyword whose value is copied
    verbatim into the header, so callers must pass the full ``"Bearer <token>"``
    string themselves. Returns ``None`` when no token is configured (the call is
    then made unauthenticated, exactly as before).
    """
    if not access_token:
        return None
    token = access_token.strip()
    if token.lower().startswith("bearer "):
        return token
    return f"Bearer {token}"


def build_firewall_api_client(
    host: Optional[str] = None,
    access_token: Optional[str] = None,
    **configuration_kwargs: Any,
) -> Any:
    """Return a configured ``sami_firewall_client.ApiClient``.

    :param host: Base URL of the firewall service (defaults to the value baked
        into the generated client, ``http://localhost``).
    :param access_token: Bearer token used for the ``HTTPBearer`` auth scheme.
    :param configuration_kwargs: Any other ``Configuration`` keyword argument
        (``verify_ssl``, ``proxy``, ``ssl_ca_cert`` ...).
    """
    sami_firewall_client = _import_firewall()
    config = sami_firewall_client.Configuration(
        host=host,
        access_token=access_token,
        **configuration_kwargs,
    )
    return sami_firewall_client.ApiClient(config)


def build_rag_api_client(
    host: Optional[str] = None,
    access_token: Optional[str] = None,
    **configuration_kwargs: Any,
) -> Any:
    """Return a configured ``sami_rag_client.ApiClient``.

    :param host: Base URL of the RAG service (defaults to ``/rag-defender``).
    :param access_token: Optional bearer token forwarded as the
        ``Authorization`` header where the endpoint accepts it.
    :param configuration_kwargs: Any other ``Configuration`` keyword argument.
    """
    sami_rag_client = _import_rag()
    config = sami_rag_client.Configuration(
        host=host,
        access_token=access_token,
        **configuration_kwargs,
    )
    return sami_rag_client.ApiClient(config)


def build_orchestrator_api(
    host: Optional[str] = None,
    access_token: Optional[str] = None,
    **configuration_kwargs: Any,
) -> Any:
    """Return a ``sami_rag_client.ORCHESTRATORApi`` (``POST /v1/rag/query``)."""
    sami_rag_client = _import_rag()
    return sami_rag_client.ORCHESTRATORApi(
        build_rag_api_client(
            host=host, access_token=access_token, **configuration_kwargs
        )
    )


def build_sami_api(
    host: Optional[str] = None,
    access_token: Optional[str] = None,
    **configuration_kwargs: Any,
) -> Any:
    """Return a ``sami_rag_client.SAMIApi`` (ingest / quarantine endpoints)."""
    sami_rag_client = _import_rag()
    return sami_rag_client.SAMIApi(
        build_rag_api_client(
            host=host, access_token=access_token, **configuration_kwargs
        )
    )


def build_rag_query_request(
    query: str,
    *,
    top_k: Optional[int] = None,
    channel: Optional[str] = None,
    retriever_backend: Optional[str] = None,
    incident_id: Optional[str] = None,
) -> Any:
    """Build a ``RagQueryRequest`` for ``POST /v1/rag/query``.

    :param incident_id: Existing firewall incident to update. The service gives
        the ``X-Incident-ID`` header precedence when both are supplied.
    """
    sami_rag_client = _import_rag()
    return sami_rag_client.RagQueryRequest(
        query=query,
        top_k=top_k,
        channel=channel,
        retriever_backend=retriever_backend,
        incident_id=incident_id,
    )


def rag_call_kwargs(
    *,
    access_token: Optional[str] = None,
    request_id: Optional[str] = None,
    incident_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    request_timeout: Optional[float] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    header_params: bool = True,
) -> Dict[str, Any]:
    """Build the per-call keyword arguments shared by the RAG SDK methods.

    :param request_id: Sent as ``X-Request-ID`` (``rag_query`` only).
    :param incident_id: Sent as ``X-Incident-ID`` (``rag_query`` only); takes
        precedence over ``RagQueryRequest.incident_id``.
    :param tenant_id: Sent as ``X-Tenant-Id`` via ``_headers``. It is not part of
        the generated signature, so it is injected as a raw header.
    :param header_params: Set to ``False`` for endpoints whose generated method
        does not accept ``x_request_id`` / ``x_incident_id`` (everything on
        ``SAMIApi``).
    """
    kwargs: Dict[str, Any] = {}

    auth = bearer_header(access_token)
    if auth is not None:
        kwargs["authorization"] = auth
    if header_params:
        if request_id is not None:
            kwargs["x_request_id"] = request_id
        if incident_id is not None:
            kwargs["x_incident_id"] = incident_id

    headers = dict(extra_headers or {})
    if tenant_id:
        headers.setdefault("X-Tenant-Id", tenant_id)
    if headers:
        kwargs["_headers"] = headers

    if request_timeout is not None:
        kwargs["_request_timeout"] = request_timeout
    return kwargs
