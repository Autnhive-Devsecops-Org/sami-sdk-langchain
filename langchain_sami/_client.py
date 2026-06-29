"""Helpers for constructing the underlying SAMI OpenAPI clients.

The two generated SDKs are installed separately:

    # RAG client  (branch: SDK_RAG_TEMP_ORPHAN)
    pip install "git+https://github.com/Autnhive-Devsecops-Org/sami-sdk-python.git@SDK_RAG_TEMP_ORPHAN"

    # Firewall client  (branch: master)
    pip install "git+https://github.com/Autnhive-Devsecops-Org/sami-sdk-python.git"

This module imports them lazily so that, for example, a project that only uses
the firewall chat model does not need the RAG client installed and vice versa.
"""

from __future__ import annotations

from typing import Any, Optional

_FIREWALL_INSTALL = (
    'pip install "git+https://github.com/Autnhive-Devsecops-Org/'
    'sami-sdk-python.git"'
)
_RAG_INSTALL = (
    'pip install "git+https://github.com/Autnhive-Devsecops-Org/'
    'sami-sdk-python.git@SDK_RAG_TEMP_ORPHAN"'
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
