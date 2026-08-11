"""Reusable Qwen embedding adapter for indexing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from typing_extensions import override

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase

DEFAULT_MODEL_NAME: Final = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_DIMENSION: Final = 512
DEFAULT_BATCH_SIZE: Final = 8


@dataclass(frozen=True, slots=True)
class UnexpectedEmbeddingOutputError(Exception):
    output_type: type[object]

    @override
    def __str__(self) -> str:
        return f"SentenceTransformer returned {self.output_type.__name__}, expected ndarray"


class QwenEmbedder:
    model_name: str
    dimension: int
    batch_size: int
    _model: SentenceTransformer

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self._model = SentenceTransformer(
            model_name,
            device=device,
            truncate_dim=dimension,
            local_files_only=local_files_only,
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer: object = self._model.tokenizer
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError("SentenceTransformer has no Hugging Face tokenizer")
        return tokenizer

    @property
    def normalized(self) -> bool:
        return True

    def embed_documents(self, texts: tuple[str, ...]) -> NDArray[np.float32]:
        output = self._model.encode_document(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self._as_float32_array(output)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        output = self._model.encode_query(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self._as_float32_array(output)

    @staticmethod
    def _as_float32_array(output: object) -> NDArray[np.float32]:
        if not isinstance(output, np.ndarray):
            raise UnexpectedEmbeddingOutputError(type(output))
        return np.asarray(output, dtype=np.float32)
