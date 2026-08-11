"""Structured JSONL audit logging for RAG questions and answers."""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .rag_contracts import RagAnswer

MINIMUM_MEANINGFUL_ANSWER_LENGTH = 20


class AnswerService(Protocol):
    async def answer(self, question: str) -> RagAnswer: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class JsonlQueryLogger:
    path: Path

    def log_answer(self, question: str, answer: RagAnswer) -> None:
        cited_sources = tuple(
            dict.fromkeys(
                f"{source.source_path}#chunk-{source.chunk_number}"
                for source in answer.sources
            )
        )
        retrieved_sources = tuple(
            dict.fromkeys(
                f"{source.source_path}#chunk-{source.chunk_number}"
                for source in answer.retrieved_sources
            )
        )
        successful = bool(
            not answer.refused
            and cited_sources
            and len(answer.text.strip()) >= MINIMUM_MEANINGFUL_ANSWER_LENGTH
        )
        self._append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": question.strip(),
                "result": answer.text,
                "chunks_found": answer.retrieved_chunks > 0,
                "retrieved_chunks": answer.retrieved_chunks,
                "response_length": len(answer.text),
                "successful_answer": successful,
                "sources": retrieved_sources,
                "cited_sources": cited_sources,
                "status": "refused" if answer.refused else "answered",
                "error_type": None,
            }
        )

    def log_error(self, question: str, error: Exception) -> None:
        self._append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": question.strip(),
                "result": "",
                "chunks_found": False,
                "retrieved_chunks": 0,
                "response_length": 0,
                "successful_answer": False,
                "sources": (),
                "cited_sources": (),
                "status": "error",
                "error_type": type(error).__name__,
            }
        )

    def _append(self, record: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                _ = stream.write(serialized + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class LoggingRagService:
    """Add durable query analytics without coupling transports to file I/O."""

    def __init__(self, service: AnswerService, logger: JsonlQueryLogger) -> None:
        self._service: AnswerService = service
        self._logger: JsonlQueryLogger = logger

    async def answer(self, question: str) -> RagAnswer:
        try:
            answer = await self._service.answer(question)
        except Exception as error:
            self._logger.log_error(question, error)
            raise
        self._logger.log_answer(question, answer)
        return answer

    async def aclose(self) -> None:
        await self._service.aclose()


__all__ = [
    "MINIMUM_MEANINGFUL_ANSWER_LENGTH",
    "AnswerService",
    "JsonlQueryLogger",
    "LoggingRagService",
]
