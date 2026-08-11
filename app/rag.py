"""Application service that orchestrates retrieval and grounded generation."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Final

from typing_extensions import override

from .prompts import UNKNOWN_ANSWER, build_messages
from .rag_contracts import (
    ChatModel,
    RagAnswer,
    RagConfig,
    RetrievedChunk,
    Retriever,
    SecurityMode,
    SecurityReport,
)
from .security import secure_chunks, uses_protected_prompt

_CITATION_PATTERN: Final = re.compile(r"\[S(\d+)]")


@dataclass(frozen=True, slots=True)
class InvalidQuestionError(ValueError):
    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


class RagService:
    """Run one bounded RAG request at a time for a small local machine."""

    _retriever: Retriever
    _chat_model: ChatModel
    _config: RagConfig
    _security_mode: SecurityMode
    _request_lock: asyncio.Semaphore

    def __init__(
        self,
        retriever: Retriever,
        chat_model: ChatModel,
        *,
        config: RagConfig | None = None,
        security_mode: SecurityMode = SecurityMode.DEFENSE_IN_DEPTH,
    ) -> None:
        self._retriever = retriever
        self._chat_model = chat_model
        self._config = config or RagConfig()
        self._security_mode = security_mode
        self._validate_config(self._config)
        self._request_lock = asyncio.Semaphore(1)

    async def answer(self, question: str) -> RagAnswer:
        normalized_question = question.strip()
        self._validate_question(normalized_question)

        async with self._request_lock:
            chunks = await asyncio.to_thread(
                self._retriever.retrieve,
                normalized_question,
                limit=self._config.top_k,
            )
            secured = secure_chunks(chunks, mode=self._security_mode)
            if not self._has_reliable_context(secured.chunks):
                return self._unknown_answer(
                    secured.report,
                    retrieved_chunks=len(chunks),
                    retrieved_sources=chunks,
                )

            messages = build_messages(
                normalized_question,
                secured.chunks,
                max_context_chars=self._config.max_context_chars,
                protect_context=uses_protected_prompt(self._security_mode),
            )
            generated = (await self._chat_model.generate(messages)).strip()
            return self._validate_and_format(
                generated,
                secured.chunks,
                secured.report,
                retrieved_chunks=len(chunks),
                retrieved_sources=chunks,
            )

    async def aclose(self) -> None:
        await self._chat_model.aclose()

    def _validate_question(self, question: str) -> None:
        if not question:
            raise InvalidQuestionError("Вопрос не должен быть пустым.")
        if len(question) > self._config.max_question_chars:
            raise InvalidQuestionError(
                f"Вопрос не должен превышать {self._config.max_question_chars} символов."
            )

    def _has_reliable_context(self, chunks: tuple[RetrievedChunk, ...]) -> bool:
        if not chunks:
            return False
        threshold = self._config.score_threshold
        return threshold is None or chunks[0].score >= threshold

    @staticmethod
    def _unknown_answer(
        report: SecurityReport | None = None,
        *,
        retrieved_chunks: int = 0,
        retrieved_sources: tuple[RetrievedChunk, ...] = (),
    ) -> RagAnswer:
        return RagAnswer(
            text=UNKNOWN_ANSWER,
            sources=(),
            refused=True,
            security=report,
            retrieved_chunks=retrieved_chunks,
            retrieved_sources=retrieved_sources,
        )

    @classmethod
    def _validate_and_format(
        cls,
        generated: str,
        chunks: tuple[RetrievedChunk, ...],
        report: SecurityReport,
        *,
        retrieved_chunks: int,
        retrieved_sources: tuple[RetrievedChunk, ...],
    ) -> RagAnswer:
        if not generated or generated.startswith(UNKNOWN_ANSWER):
            return cls._unknown_answer(
                report,
                retrieved_chunks=retrieved_chunks,
                retrieved_sources=retrieved_sources,
            )

        source_numbers = tuple(
            dict.fromkeys(int(match) for match in _CITATION_PATTERN.findall(generated))
        )
        if not source_numbers or any(
            number < 1 or number > len(chunks) for number in source_numbers
        ):
            return cls._unknown_answer(
                report,
                retrieved_chunks=retrieved_chunks,
                retrieved_sources=retrieved_sources,
            )

        cited_chunks = tuple(chunks[number - 1] for number in source_numbers)
        source_lines = tuple(
            f"- [S{number}] {chunk.source_path}#чанк-{chunk.chunk_number}"
            for number, chunk in zip(source_numbers, cited_chunks, strict=True)
        )
        text = f"{generated}\n\nИсточники:\n" + "\n".join(source_lines)
        return RagAnswer(
            text=text,
            sources=cited_chunks,
            refused=False,
            security=report,
            retrieved_chunks=retrieved_chunks,
            retrieved_sources=retrieved_sources,
        )

    @staticmethod
    def _validate_config(config: RagConfig) -> None:
        if config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if config.max_question_chars <= 0:
            raise ValueError("max_question_chars must be positive")
        if config.max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
