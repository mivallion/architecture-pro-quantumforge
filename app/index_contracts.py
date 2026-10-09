"""Shared indexing contracts and value objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override


class DocumentEmbedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def normalized(self) -> bool: ...

    def embed_documents(self, texts: tuple[str, ...]) -> NDArray[np.float32]: ...


@dataclass(frozen=True, slots=True)
class IndexConfig:
    knowledge_base_dir: Path
    database_path: Path
    faiss_path: Path


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    start_index: int


@dataclass(frozen=True, slots=True)
class BuildResult:
    scanned_documents: int
    indexed_documents: int
    indexed_chunks: int
    unchanged_documents: int
    deleted_documents: int
    total_chunks: int
    index_changed: bool
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class KnowledgeBaseNotFoundError(Exception):
    path: Path

    @override
    def __str__(self) -> str:
        return f"knowledge base directory does not exist: {self.path}"


@dataclass(frozen=True, slots=True)
class InvalidChunkError(Exception):
    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class InvalidEmbeddingError(Exception):
    expected_shape: tuple[int, int]
    actual_shape: tuple[int, ...]

    @override
    def __str__(self) -> str:
        return (
            f"expected embedding matrix {self.expected_shape}, got {self.actual_shape}"
        )
