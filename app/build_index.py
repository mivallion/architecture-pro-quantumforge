"""Build the local vector index from the Task 2 knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from .embeddings import QwenEmbedder
from .indexing import IndexConfig, RecursiveChunker, build_index

CHUNK_SIZE: Final = 500
CHUNK_OVERLAP: Final = 50


def main() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    task3_dir = repository_root / "Task3"
    embedder = QwenEmbedder()
    chunker = RecursiveChunker.from_huggingface_tokenizer(
        embedder.tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    result = build_index(
        IndexConfig(
            knowledge_base_dir=repository_root / "Task2" / "knowledge_base",
            database_path=task3_dir / "index" / "metadata.sqlite3",
            faiss_path=task3_dir / "index" / "faiss.index",
        ),
        embedder,
        chunker,
    )
    message = " ".join(
        (
            f"Indexed {result.indexed_documents}/{result.scanned_documents} documents",
            f"({result.indexed_chunks} generated chunks)",
            f"into {result.total_chunks} total chunks in {result.duration_seconds:.2f}s",
            f"({result.unchanged_documents} unchanged,",
            f"{result.deleted_documents} deleted,",
            f"changed={result.index_changed}).",
        )
    )
    print(message)


if __name__ == "__main__":
    main()
