"""Safely run the incremental index update from cron or another scheduler."""

from __future__ import annotations

import argparse
import fcntl
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .build_index import CHUNK_OVERLAP, CHUNK_SIZE
from .embeddings import QwenEmbedder
from .index_contracts import BuildResult, IndexConfig
from .index_validation import IndexValidation, IndexValidationError, validate_index
from .indexing import RecursiveChunker, build_index
from .launchd_control import restart_launchd_bot, validate_launchd_label
from .update_logging import JsonlEventWriter

DEFAULT_MAX_ATTEMPTS: Final = 3
DEFAULT_RETRY_DELAY_SECONDS: Final = 5.0


@dataclass(frozen=True, slots=True)
class UpdateConfig:
    repository_root: Path
    log_path: Path
    lock_path: Path
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS
    launchd_label: str | None = None

    @property
    def knowledge_base_dir(self) -> Path:
        return self.repository_root / "Task2" / "knowledge_base"

    @property
    def database_path(self) -> Path:
        return self.repository_root / "Task3" / "index" / "metadata.sqlite3"

    @property
    def faiss_path(self) -> Path:
        return self.repository_root / "Task3" / "index" / "faiss.index"


def run_update(
    config: UpdateConfig,
    *,
    build_operation: Callable[[], BuildResult] | None = None,
    validation_operation: Callable[[], IndexValidation] | None = None,
    restart_operation: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    _validate_config(config)
    writer = JsonlEventWriter(config.log_path)
    config.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with config.lock_path.open("a", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            writer.write("update_skipped", reason="already_running")
            return 0

        writer.write(
            "update_started",
            source=str(config.knowledge_base_dir),
            max_attempts=config.max_attempts,
        )
        started = time.perf_counter()
        build_action = build_operation or (lambda: _build(config))
        validate_action = validation_operation or (
            lambda: validate_index(config.database_path, config.faiss_path)
        )
        result: BuildResult | None = None
        validation: IndexValidation | None = None
        changed_during_run = False
        indexed_documents = 0
        indexed_chunks = 0
        deleted_documents = 0

        for attempt in range(1, config.max_attempts + 1):
            try:
                result = build_action()
                changed_during_run = changed_during_run or result.index_changed
                indexed_documents += result.indexed_documents
                indexed_chunks += result.indexed_chunks
                deleted_documents += result.deleted_documents
                validation = validate_action()
                break
            except Exception as error:  # noqa: BLE001 - retry boundary for external operations.
                writer.write(
                    "update_attempt_failed",
                    attempt=attempt,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                if attempt == config.max_attempts:
                    writer.write(
                        "update_failed",
                        attempts=attempt,
                        duration_seconds=round(time.perf_counter() - started, 3),
                        errors=attempt,
                    )
                    return 1
                writer.write(
                    "retry_scheduled",
                    attempt=attempt + 1,
                    delay_seconds=config.retry_delay_seconds,
                )
                sleep(config.retry_delay_seconds)

        if result is None or validation is None:
            raise RuntimeError("update completed without a result")

        restarted = False
        if changed_during_run and (restart_operation or config.launchd_label):
            try:
                if restart_operation is not None:
                    restart_operation()
                elif config.launchd_label is not None:
                    restart_launchd_bot(config.launchd_label)
                restarted = True
            except Exception as error:  # noqa: BLE001 - scheduler must log restart failures.
                writer.write(
                    "bot_restart_failed",
                    error_type=type(error).__name__,
                    error=str(error),
                )
                return 1

        writer.write(
            "update_completed",
            duration_seconds=round(time.perf_counter() - started, 3),
            indexed_documents=indexed_documents,
            indexed_chunks=indexed_chunks,
            unchanged_documents=result.unchanged_documents,
            deleted_documents=deleted_documents,
            total_documents=validation.documents,
            total_chunks=validation.chunks,
            index_dimension=validation.dimension,
            index_size_bytes=validation.index_size_bytes,
            index_changed=changed_during_run,
            bot_restarted=restarted,
            errors=0,
        )
        return 0


def _build(config: UpdateConfig) -> BuildResult:
    embedder = QwenEmbedder(local_files_only=True)
    chunker = RecursiveChunker.from_huggingface_tokenizer(
        embedder.tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return build_index(
        IndexConfig(
            knowledge_base_dir=config.knowledge_base_dir,
            database_path=config.database_path,
            faiss_path=config.faiss_path,
        ),
        embedder,
        chunker,
    )


def _validate_config(config: UpdateConfig) -> None:
    if config.max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if config.retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must not be negative")
    if config.launchd_label is not None:
        validate_launchd_label(config.launchd_label)


def _parse_arguments(arguments: Sequence[str] | None = None) -> UpdateConfig:
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--repository-root", type=Path, default=repository_root)
    _ = parser.add_argument("--log-file", type=Path)
    _ = parser.add_argument("--lock-file", type=Path)
    _ = parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    _ = parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=DEFAULT_RETRY_DELAY_SECONDS,
    )
    _ = parser.add_argument("--launchd-label")
    namespace = parser.parse_args(arguments)
    root = namespace.repository_root.resolve()
    return UpdateConfig(
        repository_root=root,
        log_path=(namespace.log_file or root / "Task6" / "logs" / "index-update.jsonl"),
        lock_path=(namespace.lock_file or root / "Task6" / "run" / "update.lock"),
        max_attempts=namespace.max_attempts,
        retry_delay_seconds=namespace.retry_delay_seconds,
        launchd_label=namespace.launchd_label,
    )


def main(arguments: Sequence[str] | None = None) -> None:
    raise SystemExit(run_update(_parse_arguments(arguments)))


if __name__ == "__main__":
    main()


__all__ = [
    "IndexValidation",
    "IndexValidationError",
    "JsonlEventWriter",
    "UpdateConfig",
    "restart_launchd_bot",
    "run_update",
    "validate_index",
]
