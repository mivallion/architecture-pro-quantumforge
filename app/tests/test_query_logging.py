from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.query_logging import JsonlQueryLogger, LoggingRagService
from app.rag_contracts import RagAnswer, RetrievedChunk


class FakeService:
    def __init__(
        self, *, answer: RagAnswer | None = None, error: Exception | None = None
    ) -> None:
        self._answer: RagAnswer | None = answer
        self._error: Exception | None = error
        self.closed: bool = False

    async def answer(self, question: str) -> RagAnswer:
        _ = question
        if self._error is not None:
            raise self._error
        if self._answer is None:
            raise RuntimeError("fake service has no answer")
        return self._answer

    async def aclose(self) -> None:
        self.closed = True


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1,
        score=0.8,
        title="Pathkeeper",
        source_path="19-pathkeeper.md",
        chunk_number=0,
        start_index=0,
        text="Pathkeeper explains recipes.",
    )


def _read_log(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8").strip())


@pytest.mark.asyncio
async def test_logging_service_records_answer_and_sources(tmp_path: Path) -> None:
    # Given
    answer = RagAnswer(
        text="Pathkeeper explains all crafting recipes. [S1]",
        sources=(_chunk(),),
        refused=False,
        retrieved_chunks=4,
        retrieved_sources=(_chunk(),),
    )
    log_path = tmp_path / "logs.jsonl"
    service = LoggingRagService(FakeService(answer=answer), JsonlQueryLogger(log_path))

    # When
    actual = await service.answer(" What does Pathkeeper do? ")

    # Then
    assert actual is answer
    record = _read_log(log_path)
    assert record["query"] == "What does Pathkeeper do?"
    assert record["chunks_found"] is True
    assert record["retrieved_chunks"] == 4
    assert record["successful_answer"] is True
    assert record["sources"] == ["19-pathkeeper.md#chunk-0"]
    assert record["cited_sources"] == ["19-pathkeeper.md#chunk-0"]
    assert record["status"] == "answered"


@pytest.mark.asyncio
async def test_logging_service_records_refusal(tmp_path: Path) -> None:
    # Given
    answer = RagAnswer(
        text="Я не знаю: в базе знаний недостаточно информации.",
        sources=(),
        refused=True,
        retrieved_chunks=4,
        retrieved_sources=(_chunk(),),
    )
    log_path = tmp_path / "logs.jsonl"
    service = LoggingRagService(FakeService(answer=answer), JsonlQueryLogger(log_path))

    # When
    _ = await service.answer("Unknown topic")

    # Then
    record = _read_log(log_path)
    assert record["chunks_found"] is True
    assert record["sources"] == ["19-pathkeeper.md#chunk-0"]
    assert record["cited_sources"] == []
    assert record["successful_answer"] is False
    assert record["status"] == "refused"


@pytest.mark.asyncio
async def test_logging_service_records_and_reraises_error(tmp_path: Path) -> None:
    # Given
    log_path = tmp_path / "logs.jsonl"
    service = LoggingRagService(
        FakeService(error=RuntimeError("internal detail")),
        JsonlQueryLogger(log_path),
    )

    # When / Then
    with pytest.raises(RuntimeError, match="internal detail"):
        _ = await service.answer("Question")

    record = _read_log(log_path)
    assert record["status"] == "error"
    assert record["error_type"] == "RuntimeError"
    assert record["result"] == ""
    assert "internal detail" not in log_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_logging_service_closes_wrapped_service(tmp_path: Path) -> None:
    # Given
    wrapped_service = FakeService()
    service = LoggingRagService(
        wrapped_service, JsonlQueryLogger(tmp_path / "logs.jsonl")
    )

    # When
    await service.aclose()

    # Then
    assert wrapped_service.closed is True
