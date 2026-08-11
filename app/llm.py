"""Async Ollama adapter used by the RAG service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from ollama import AsyncClient, ResponseError

from .rag_contracts import ChatMessage


class OllamaChatError(RuntimeError):
    """A safe, user-facing failure while communicating with Ollama."""


class _OllamaMessage(Protocol):
    content: str | None


class _OllamaResponse(Protocol):
    message: _OllamaMessage


class _OllamaClient(Protocol):
    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, object]],
        think: bool,
        options: Mapping[str, object],
        keep_alive: str,
    ) -> _OllamaResponse: ...

    async def close(self) -> None: ...


class OllamaChatModel:
    """One reusable Ollama client with conservative local-generation settings."""

    def __init__(
        self,
        *,
        model: str,
        host: str | None = None,
        keep_alive: str = "0",
        temperature: float = 0.1,
        num_ctx: int = 2_048,
        num_predict: int = 384,
        client: _OllamaClient | None = None,
    ) -> None:
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("Ollama model name must not be empty.")
        if not keep_alive.strip():
            raise ValueError("Ollama keep_alive must not be empty.")
        if num_ctx <= 0 or num_predict <= 0:
            raise ValueError("Ollama token limits must be positive.")

        self._model = normalized_model
        self._keep_alive = keep_alive
        self._options: dict[str, object] = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }
        self._client = client or cast(_OllamaClient, AsyncClient(host=host))

    async def generate(self, messages: tuple[ChatMessage, ...]) -> str:
        """Generate a grounded answer without requesting Ollama's thinking output."""
        if not messages:
            raise ValueError("At least one chat message is required.")

        request_messages: list[Mapping[str, object]] = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]
        try:
            response = await self._client.chat(
                model=self._model,
                messages=request_messages,
                think=False,
                options=self._options,
                keep_alive=self._keep_alive,
            )
        except (ConnectionError, ResponseError) as error:
            raise OllamaChatError("Ollama is unavailable or rejected the generation request.") from error

        content = response.message.content
        if not isinstance(content, str) or not content.strip():
            raise OllamaChatError("Ollama returned an empty textual response.")
        return content.strip()

    async def aclose(self) -> None:
        """Release the HTTP client resources during application shutdown."""
        await self._client.close()
