"""Tokenizer-aware document chunking."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter
from transformers import PreTrainedTokenizerBase

from .index_contracts import InvalidChunkError, TextChunk


@dataclass(frozen=True, slots=True)
class RecursiveChunker:
    splitter: TextSplitter
    chunk_size: int
    chunk_overlap: int
    strategy: str = "recursive-character-hf-tokenizer-v2"

    @classmethod
    def from_huggingface_tokenizer(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> RecursiveChunker:
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return cls(splitter, chunk_size, chunk_overlap)

    def split(self, text: str) -> tuple[TextChunk, ...]:
        documents = self.splitter.create_documents([text])
        chunks: list[TextChunk] = []
        for document in documents:
            start_index = document.metadata.get("start_index")
            if not isinstance(start_index, int):
                raise InvalidChunkError(
                    "splitter did not return an integer start_index"
                )
            chunks.append(TextChunk(document.page_content, start_index))
        return tuple(chunks)
