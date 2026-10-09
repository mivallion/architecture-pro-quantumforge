"""Composition root for the reusable local RAG service."""

from __future__ import annotations

from pathlib import Path

from .embeddings import QwenEmbedder
from .llm import OllamaChatModel
from .query_logging import JsonlQueryLogger, LoggingRagService
from .rag import RagService
from .rag_contracts import RagConfig, SecurityMode
from .retrieval import FaissRetriever
from .settings import AppSettings


def create_rag_service(
    settings: AppSettings | None = None,
    *,
    query_log_path: Path | None = None,
) -> LoggingRagService:
    resolved = settings or AppSettings.from_environment()
    if not resolved.faiss_path.is_file():
        raise FileNotFoundError(f"FAISS index not found: {resolved.faiss_path}")
    if not resolved.database_path.is_file():
        raise FileNotFoundError(
            f"metadata database not found: {resolved.database_path}"
        )

    embedder = QwenEmbedder(local_files_only=resolved.embedding_local_files_only)
    retriever = FaissRetriever(
        embedder,
        faiss_path=resolved.faiss_path,
        database_path=resolved.database_path,
    )
    model = OllamaChatModel(
        model=resolved.ollama_model,
        host=resolved.ollama_host,
        keep_alive=resolved.ollama_keep_alive,
    )
    service = RagService(
        retriever,
        model,
        security_mode=SecurityMode.DEFENSE_IN_DEPTH,
        config=RagConfig(
            top_k=resolved.top_k,
            score_threshold=resolved.score_threshold,
            max_question_chars=resolved.max_question_chars,
            max_context_chars=resolved.max_context_chars,
        ),
    )
    return LoggingRagService(
        service,
        JsonlQueryLogger(
            query_log_path or resolved.repository_root / "Task7" / "logs.jsonl"
        ),
    )
