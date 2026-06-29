"""LangChain chat model backed by the SAMI AI Firewall.

``ChatSamiFirewall`` sends the conversation through the firewall's text prompt
endpoint (``POST /ai-firewall/firewall/v1/prompt/text``). The firewall inspects
and sanitises the prompt, forwards it to the configured LLM provider, and
returns the (guarded) completion. From the caller's point of view it behaves
like any other LangChain chat model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from ._client import build_firewall_api_client
from ._utils import extract_assistant_text, lc_message_to_role


class ChatSamiFirewall(BaseChatModel):
    """Chat model that routes completions through the SAMI AI Firewall.

    Example:
        .. code-block:: python

            from langchain_sami import ChatSamiFirewall

            llm = ChatSamiFirewall(
                host="https://sami.example.com",
                access_token="...",          # HTTPBearer token
                ai_provider="openai",
                ai_key="sk-...",
            )
            llm.invoke("Summarise our refund policy.")
    """

    host: Optional[str] = None
    """Base URL of the firewall service. Defaults to the generated client value."""

    access_token: Optional[str] = None
    """Bearer token for the firewall's ``HTTPBearer`` auth scheme."""

    ai_key: Optional[str] = None
    """Optional explicit provider API key (sent as ``AI_KEY``)."""

    ai_url: Optional[str] = None
    """Optional custom provider base URL (sent as ``AI_URL``)."""

    ai_provider: Optional[str] = None
    """Optional provider routing string (sent as ``AI_PROVIDER``)."""

    request_timeout: Optional[float] = Field(default=None, alias="timeout")
    """Per-request timeout in seconds passed to the underlying client."""

    client_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Extra ``Configuration`` keyword arguments (``verify_ssl``, ``proxy`` ...)."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    _api_client: Any = PrivateAttr(default=None)
    _chat_api: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "sami-firewall"

    def _get_chat_api(self) -> Any:
        if self._chat_api is None:
            import sami_firewall_client  # local import; raises a clear error if missing

            self._api_client = build_firewall_api_client(
                host=self.host,
                access_token=self.access_token,
                **self.client_kwargs,
            )
            self._chat_api = sami_firewall_client.ChatApi(self._api_client)
        return self._chat_api

    def _build_request(self, messages: List[BaseMessage]) -> Any:
        import sami_firewall_client

        chat_messages = [
            sami_firewall_client.ChatMessage(
                role=lc_message_to_role(m),
                content=m.content if isinstance(m.content, str) else str(m.content),
            )
            for m in messages
        ]
        return sami_firewall_client.ChatCompletionRequest(
            messages=chat_messages,
            AI_KEY=self.ai_key,
            AI_URL=self.ai_url,
            AI_PROVIDER=self.ai_provider,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        chat_api = self._get_chat_api()
        request = self._build_request(messages)

        call_kwargs: Dict[str, Any] = {}
        if self.request_timeout is not None:
            call_kwargs["_request_timeout"] = self.request_timeout

        raw = chat_api.adapter_chat(request, **call_kwargs)
        text = extract_assistant_text(raw)

        message = AIMessage(
            content=text,
            response_metadata={"raw": raw, "model_name": self._llm_type},
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation], llm_output={"raw": raw})

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "ai_provider": self.ai_provider,
            "ai_url": self.ai_url,
        }
