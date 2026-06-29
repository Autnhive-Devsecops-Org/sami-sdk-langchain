"""Conversion helpers shared across the LangChain integrations."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# LangChain message class -> the ``role`` string the firewall expects.
_ROLE_BY_TYPE = {
    HumanMessage: "user",
    AIMessage: "assistant",
    SystemMessage: "system",
    ToolMessage: "tool",
    FunctionMessage: "function",
}


def lc_message_to_role(message: BaseMessage) -> str:
    """Map a LangChain message to a SAMI/OpenAI-style ``role`` string."""
    if isinstance(message, ChatMessage):
        return message.role
    for cls, role in _ROLE_BY_TYPE.items():
        if isinstance(message, cls):
            return role
    # Fall back to the message ``type`` (``human``, ``ai`` ...).
    return message.type


def lc_messages_to_dicts(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Convert LangChain messages to ``{"role", "content"}`` dictionaries."""
    payload: List[Dict[str, str]] = []
    for message in messages:
        content = message.content
        if not isinstance(content, str):
            # Multi-part content -> flatten to text for the text endpoint.
            content = json.dumps(content)
        payload.append({"role": lc_message_to_role(message), "content": content})
    return payload


def extract_assistant_text(response: Any) -> str:
    """Best-effort extraction of assistant text from a firewall response.

    The firewall ``adapter_chat`` endpoint is typed as ``Dict[str, object]`` so
    the exact shape is provider dependent. We probe the common shapes (OpenAI
    ``choices``, plain ``content``/``answer``/``response`` keys) and fall back to
    a JSON dump so no content is silently lost.
    """
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        # Pydantic model or similar - try to coerce to a dict.
        for attr in ("model_dump", "to_dict", "dict"):
            method = getattr(response, attr, None)
            if callable(method):
                try:
                    response = method()
                    break
                except Exception:  # pragma: no cover - defensive
                    pass
    if not isinstance(response, dict):
        return str(response)

    # OpenAI-style chat completion.
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                return str(message["content"])
            if first.get("text") is not None:
                return str(first["text"])

    # Flat shapes.
    for key in ("content", "answer", "response", "output", "text", "result"):
        value = response.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("content"), str):
            return value["content"]

    # Nothing matched - return the raw payload so callers can inspect it.
    return json.dumps(response)


def to_serializable(value: Any) -> Any:
    """Coerce a generated pydantic model (or anything) into a plain structure."""
    for attr in ("model_dump", "to_dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:  # pragma: no cover - defensive
                pass
    return value
