"""LangChain integrations for the SAMI AI Firewall and RAG services.

Components
----------
* :class:`ChatSamiFirewall`        - chat model guarded by the AI Firewall.
* :class:`SamiRagRetriever`        - retriever backed by the RAG orchestrator.
* :class:`SamiRagChain`            - one-shot end-to-end RAG runnable.
* :func:`make_sami_rag_tools`      - agent tools for ingest / quarantine review.
"""

from ._client import build_firewall_api_client, build_rag_api_client
from .chat_models import ChatSamiFirewall
from .rag import SamiRagAnswer, SamiRagChain
from .retrievers import SamiRagRetriever
from .tools import make_sami_rag_tools

__all__ = [
    "ChatSamiFirewall",
    "SamiRagRetriever",
    "SamiRagChain",
    "SamiRagAnswer",
    "make_sami_rag_tools",
    "build_firewall_api_client",
    "build_rag_api_client",
]

__version__ = "0.1.0"
