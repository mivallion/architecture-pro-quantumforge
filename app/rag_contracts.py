"""Shared contracts and immutable values for the RAG application."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class SecurityMode(StrEnum):
    UNPROTECTED = "unprotected"
    PRE_PROMPT = "pre_prompt"
    SANITIZE = "sanitize"
    FILTER = "filter"
    DEFENSE_IN_DEPTH = "defense_in_depth"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: int
    score: float
    title: str
    source_path: str
    chunk_number: int
    start_index: int
    text: str


@dataclass(frozen=True, slots=True)
class SecurityReport:
    mode: SecurityMode
    detected_chunks: int = 0
    removed_chunks: int = 0
    sanitized_chunks: int = 0


@dataclass(frozen=True, slots=True)
class RagAnswer:
    text: str
    sources: tuple[RetrievedChunk, ...]
    refused: bool
    security: SecurityReport | None = None
    retrieved_chunks: int = 0
    retrieved_sources: tuple[RetrievedChunk, ...] = ()


@dataclass(frozen=True, slots=True)
class RagConfig:
    top_k: int = 4
    score_threshold: float | None = None
    max_question_chars: int = 2_000
    max_context_chars: int = 6_000


class Retriever(Protocol):
    def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedChunk, ...]: ...


class ChatModel(Protocol):
    async def generate(self, messages: tuple[ChatMessage, ...]) -> str: ...

    async def aclose(self) -> None: ...
