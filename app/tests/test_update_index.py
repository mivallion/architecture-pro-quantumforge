from __future__ import annotations

import fcntl
import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pytest

from app.index_contracts import BuildResult
from app.index_store import initialize_schema
from app.update_index import (
    IndexValidation,
    IndexValidationError,
    UpdateConfig,
    run_update,
    validate_index,
)


def _config(tmp_path: Path, *, max_attempts: int = 3) -> UpdateConfig:
    return UpdateConfig(
        repository_root=tmp_path,
        log_path=tmp_path / "logs" / "update.jsonl",
        lock_path=tmp_path / "run" / "update.lock",
        max_attempts=max_attempts,
        retry_delay_seconds=0.25,
    )


def _result(*, changed: bool = True) -> BuildResult:
    return BuildResult(
        scanned_documents=3,
        indexed_documents=1 if changed else 0,
        indexed_chunks=2 if changed else 0,
        unchanged_documents=2 if changed else 3,
        deleted_documents=0,
        total_chunks=4,
        index_changed=changed,
        duration_seconds=0.1,
    )


def _validation() -> IndexValidation:
    return IndexValidation(
        documents=3,
        chunks=4,
        dimension=512,
        index_size_bytes=8_192,
    )


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_successful_update_logs_metrics_and_restarts_changed_index(
    tmp_path: Path,
) -> None:
    # Given
    config = _config(tmp_path)
    restarts: list[bool] = []

    # When
    exit_code = run_update(
        config,
        build_operation=_result,
        validation_operation=_validation,
        restart_operation=lambda: restarts.append(True),
    )

    # Then
    assert exit_code == 0
    assert restarts == [True]
    events = _events(config.log_path)
    assert [event["event"] for event in events] == [
        "update_started",
        "update_completed",
    ]
    assert events[-1]["indexed_chunks"] == 2
    assert events[-1]["total_chunks"] == 4
    assert events[-1]["index_size_bytes"] == 8_192
    assert events[-1]["index_changed"] is True
    assert events[-1]["bot_restarted"] is True
    assert events[-1]["errors"] == 0


def test_unchanged_index_does_not_restart_bot(tmp_path: Path) -> None:
    # Given
    config = _config(tmp_path)
    restarts: list[bool] = []

    # When
    exit_code = run_update(
        config,
        build_operation=lambda: _result(changed=False),
        validation_operation=_validation,
        restart_operation=lambda: restarts.append(True),
    )

    # Then
    assert exit_code == 0
    assert restarts == []
    assert _events(config.log_path)[-1]["bot_restarted"] is False


def test_failed_attempt_is_retried_and_logged(tmp_path: Path) -> None:
    # Given
    config = _config(tmp_path)
    attempts = 0
    sleeps: list[float] = []

    def build() -> BuildResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary failure")
        return _result()

    # When
    exit_code = run_update(
        config,
        build_operation=build,
        validation_operation=_validation,
        sleep=sleeps.append,
    )

    # Then
    assert exit_code == 0
    assert attempts == 2
    assert sleeps == [0.25]
    assert [event["event"] for event in _events(config.log_path)] == [
        "update_started",
        "update_attempt_failed",
        "retry_scheduled",
        "update_completed",
    ]


def test_change_is_not_lost_when_validation_is_retried(tmp_path: Path) -> None:
    # Given
    config = _config(tmp_path)
    build_results = iter((_result(changed=True), _result(changed=False)))
    validation_attempts = 0
    restarts: list[bool] = []

    def validate() -> IndexValidation:
        nonlocal validation_attempts
        validation_attempts += 1
        if validation_attempts == 1:
            raise RuntimeError("temporary validation failure")
        return _validation()

    # When
    exit_code = run_update(
        config,
        build_operation=lambda: next(build_results),
        validation_operation=validate,
        restart_operation=lambda: restarts.append(True),
        sleep=lambda _: None,
    )

    # Then
    assert exit_code == 0
    assert restarts == [True]
    completed = _events(config.log_path)[-1]
    assert completed["index_changed"] is True
    assert completed["indexed_chunks"] == 2


def test_final_failure_returns_nonzero_and_logs_error(tmp_path: Path) -> None:
    # Given
    config = _config(tmp_path, max_attempts=2)

    def fail() -> BuildResult:
        raise RuntimeError("persistent failure")

    # When
    exit_code = run_update(
        config,
        build_operation=fail,
        validation_operation=_validation,
        sleep=lambda _: None,
    )

    # Then
    assert exit_code == 1
    events = _events(config.log_path)
    assert events[-1]["event"] == "update_failed"
    assert sum(event["event"] == "update_attempt_failed" for event in events) == 2


def test_overlapping_update_is_skipped(tmp_path: Path) -> None:
    # Given
    config = _config(tmp_path)
    config.lock_path.parent.mkdir(parents=True)
    with config.lock_path.open("a", encoding="utf-8") as active_lock:
        fcntl.flock(active_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        # When
        exit_code = run_update(
            config,
            build_operation=lambda: pytest.fail("build must not run"),
            validation_operation=_validation,
        )

    # Then
    assert exit_code == 0
    assert _events(config.log_path)[0]["event"] == "update_skipped"


def test_validate_index_matches_sqlite_and_faiss_ids(tmp_path: Path) -> None:
    # Given
    database_path = tmp_path / "metadata.sqlite3"
    faiss_path = tmp_path / "faiss.index"
    with sqlite3.connect(database_path) as connection:
        initialize_schema(connection)
        cursor = connection.execute(
            "INSERT INTO documents(source_path, title, content_hash) VALUES (?, ?, ?)",
            ("example.md", "Example", "abc"),
        )
        assert cursor.lastrowid is not None
        document_id = int(cursor.lastrowid)
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        chunk_cursor = connection.execute(
            """INSERT INTO chunks(
                   document_id, chunk_number, start_index, text, embedding
               ) VALUES (?, ?, ?, ?, ?)""",
            (document_id, 0, 0, "Example text", vector.tobytes()),
        )
        assert chunk_cursor.lastrowid is not None
        chunk_id = int(chunk_cursor.lastrowid)

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(512))
    index.add_with_ids(vector.reshape(1, -1), np.asarray([chunk_id], dtype=np.int64))
    faiss.write_index(index, str(faiss_path))

    # When
    validation = validate_index(database_path, faiss_path)

    # Then
    assert validation.documents == 1
    assert validation.chunks == 1
    assert validation.dimension == 512
    assert validation.index_size_bytes == faiss_path.stat().st_size


def test_validate_index_rejects_identifier_mismatch(tmp_path: Path) -> None:
    # Given
    database_path = tmp_path / "metadata.sqlite3"
    faiss_path = tmp_path / "faiss.index"
    with sqlite3.connect(database_path) as connection:
        initialize_schema(connection)
        cursor = connection.execute(
            "INSERT INTO documents(source_path, title, content_hash) VALUES (?, ?, ?)",
            ("example.md", "Example", "abc"),
        )
        assert cursor.lastrowid is not None
        vector = np.zeros(512, dtype=np.float32)
        _ = connection.execute(
            """INSERT INTO chunks(
                   document_id, chunk_number, start_index, text, embedding
               ) VALUES (?, ?, ?, ?, ?)""",
            (int(cursor.lastrowid), 0, 0, "Example text", vector.tobytes()),
        )

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(512))
    index.add_with_ids(vector.reshape(1, -1), np.asarray([999], dtype=np.int64))
    faiss.write_index(index, str(faiss_path))

    # When / Then
    with pytest.raises(IndexValidationError, match="identifiers"):
        _ = validate_index(database_path, faiss_path)
