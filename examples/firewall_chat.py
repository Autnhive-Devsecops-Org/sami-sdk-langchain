"""Chat through the SAMI AI Firewall using a standard LangChain chat model.

Run:
    pip install -e "../[firewall]"
    SAMI_HOST=https://sami.example.com SAMI_TOKEN=... python firewall_chat.py
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage

from langchain_sami import ChatSamiFirewall


def main() -> None:
    llm = ChatSamiFirewall(
        host=os.environ.get("SAMI_HOST"),
        access_token=os.environ.get("SAMI_TOKEN"),
        ai_provider=os.environ.get("AI_PROVIDER", "openai"),
        ai_key=os.environ.get("AI_KEY"),
    )

    messages = [
        SystemMessage(content="You are a concise support assistant."),
        HumanMessage(content="What payment methods do we accept?"),
    ]
    response = llm.invoke(messages)
    print("Answer:", response.content)
    print("Raw firewall payload:", response.response_metadata.get("raw"))


if __name__ == "__main__":
    main()
