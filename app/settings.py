"""Environment-backed settings for local RAG and Telegram entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

DEFAULT_OLLAMA_HOST: Final = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL: Final = "qwen3:4b-instruct-2507-q4_K_M"
DEFAULT_OLLAMA_KEEP_ALIVE: Final = "0"
DEFAULT_TOP_K: Final = 4
DEFAULT_MAX_QUESTION_CHARS: Final = 2_000
DEFAULT_MAX_CONTEXT_CHARS: Final = 6_000


@dataclass(frozen=True, slots=True)
class AppSettings:
    repository_root: Path
    ollama_host: str = DEFAULT_OLLAMA_HOST
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_keep_alive: str = DEFAULT_OLLAMA_KEEP_ALIVE
    embedding_local_files_only: bool = True
    top_k: int = DEFAULT_TOP_K
    score_threshold: float | None = None
    max_question_chars: int = DEFAULT_MAX_QUESTION_CHARS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS

    @property
    def faiss_path(self) -> Path:
        return self.repository_root / "Task3" / "index" / "faiss.index"

    @property
    def database_path(self) -> Path:
        return self.repository_root / "Task3" / "index" / "metadata.sqlite3"

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> AppSettings:
        values = os.environ if environment is None else environment
        default_root = Path(__file__).resolve().parent.parent
        threshold_value = values.get("RAG_SCORE_THRESHOLD", "").strip()
        repository_root = values.get("RAG_REPOSITORY_ROOT")
        return cls(
            repository_root=default_root if repository_root is None else Path(repository_root),
            ollama_host=values.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            ollama_model=values.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            ollama_keep_alive=values.get(
                "OLLAMA_KEEP_ALIVE", DEFAULT_OLLAMA_KEEP_ALIVE
            ),
            embedding_local_files_only=_parse_bool(
                values.get("EMBEDDING_LOCAL_FILES_ONLY", "true")
            ),
            top_k=int(values.get("RAG_TOP_K", str(DEFAULT_TOP_K))),
            score_threshold=float(threshold_value) if threshold_value else None,
            max_question_chars=int(
                values.get(
                    "RAG_MAX_QUESTION_CHARS", str(DEFAULT_MAX_QUESTION_CHARS)
                )
            ),
            max_context_chars=int(
                values.get("RAG_MAX_CONTEXT_CHARS", str(DEFAULT_MAX_CONTEXT_CHARS))
            ),
        )


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")
