from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from ollama import ResponseError

from app.llm import OllamaChatError, OllamaChatModel
from app.rag_contracts import ChatMessage, MessageRole


@dataclass
class FakeMessage:
    content: str | None


@dataclass
class FakeResponse:
    message: FakeMessage


class FakeOllamaClient:
    def __init__(self, response: FakeResponse | None = None) -> None:
        self.response = response or FakeResponse(FakeMessage("Grounded answer"))
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self.error: Exception | None = None

    async def chat(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    async def close(self) -> None:
        self.closed = True


def _messages() -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(MessageRole.SYSTEM, "Use only supplied context."),
        ChatMessage(MessageRole.USER, "Who is the healer?"),
    )


def test_generate_sends_expected_ollama_options_and_strips_answer() -> None:
    # Given
    client = FakeOllamaClient(FakeResponse(FakeMessage("  Grounded answer  ")))
    model = OllamaChatModel(
        model="qwen3:4b-instruct-2507-q4_K_M",
        host="http://localhost:11434",
        client=client,
    )

    # When
    answer = asyncio.run(model.generate(_messages()))

    # Then
    assert answer == "Grounded answer"
    assert client.calls == [
        {
            "model": "qwen3:4b-instruct-2507-q4_K_M",
            "messages": [
                {"role": "system", "content": "Use only supplied context."},
                {"role": "user", "content": "Who is the healer?"},
            ],
            "think": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 2_048,
                "num_predict": 384,
            },
            "keep_alive": "0",
        }
    ]


def test_generate_maps_connection_failure_to_safe_error() -> None:
    # Given
    client = FakeOllamaClient()
    client.error = ConnectionError("http://private-host:11434 refused")
    model = OllamaChatModel(model="qwen3:4b", client=client)

    # When / Then
    with pytest.raises(OllamaChatError, match="unavailable or rejected") as error:
        _ = asyncio.run(model.generate(_messages()))
    assert "private-host" not in str(error.value)


def test_generate_maps_ollama_response_failure_to_safe_error() -> None:
    # Given
    client = FakeOllamaClient()
    client.error = ResponseError("model secret-model is missing", 404)
    model = OllamaChatModel(model="qwen3:4b", client=client)

    # When / Then
    with pytest.raises(OllamaChatError, match="unavailable or rejected") as error:
        _ = asyncio.run(model.generate(_messages()))
    assert "secret-model" not in str(error.value)


def test_generate_rejects_empty_response_and_close_releases_client() -> None:
    # Given
    client = FakeOllamaClient(FakeResponse(FakeMessage("   ")))
    model = OllamaChatModel(model="qwen3:4b", client=client)

    # When / Then
    with pytest.raises(OllamaChatError, match="empty textual response"):
        _ = asyncio.run(model.generate(_messages()))

    asyncio.run(model.aclose())
    assert client.closed is True


def test_model_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="model name"):
        _ = OllamaChatModel(model="  ")
    with pytest.raises(ValueError, match="token limits"):
        _ = OllamaChatModel(model="qwen3:4b", num_ctx=0)
