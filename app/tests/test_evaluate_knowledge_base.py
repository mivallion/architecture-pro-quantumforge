from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app import evaluate_knowledge_base
from app.evaluate_knowledge_base import (
    GoldenCase,
    assess_answer,
    evaluate,
    load_golden_cases,
)
from app.rag_contracts import RagAnswer, RetrievedChunk
from app.settings import AppSettings


def _case(*, should_answer: bool = True) -> GoldenCase:
    return GoldenCase(
        case_id="case-1",
        topic="Pathkeeper",
        question="What does Pathkeeper do?",
        should_answer=should_answer,
        expected_source="19-pathkeeper.md" if should_answer else None,
        expected_answer="Pathkeeper explains recipes.",
        required_keywords=("Pathkeeper",) if should_answer else (),
    )


def _answer(*, refused: bool = False, source: str = "19-pathkeeper.md") -> RagAnswer:
    sources = (
        ()
        if refused
        else (
            RetrievedChunk(
                chunk_id=1,
                score=0.8,
                title="Pathkeeper",
                source_path=source,
                chunk_number=0,
                start_index=0,
                text="Pathkeeper explains recipes.",
            ),
        )
    )
    return RagAnswer(
        text=(
            "Я не знаю: в базе знаний недостаточно информации."
            if refused
            else "Pathkeeper explains recipes."
        ),
        sources=sources,
        refused=refused,
        retrieved_chunks=4,
        retrieved_sources=sources
        or (
            RetrievedChunk(
                chunk_id=2,
                score=0.5,
                title="Unrelated",
                source_path="unrelated.md",
                chunk_number=0,
                start_index=0,
                text="Unrelated text.",
            ),
        ),
    )


def test_known_answer_passes_with_expected_source_and_keyword() -> None:
    record = assess_answer(_case(), _answer(), duration_seconds=0.5)

    assert record.passed is True
    assert record.source_match is True
    assert record.retrieval_source_match is True
    assert record.keywords_match is True


def test_known_answer_fails_with_irrelevant_source() -> None:
    record = assess_answer(
        _case(),
        _answer(source="unrelated.md"),
        duration_seconds=0.5,
    )

    assert record.passed is False
    assert record.failure_reason == "expected_source_not_cited"


def test_known_answer_fails_with_forbidden_hallucination() -> None:
    case = GoldenCase(
        case_id="case-hallucination",
        topic="Chromist",
        question="What does Chromist trade?",
        should_answer=True,
        expected_source="19-pathkeeper.md",
        expected_answer="Animated dyes.",
        required_keywords=("Pathkeeper",),
        forbidden_keywords=("invented reward",),
    )
    answer = RagAnswer(
        text="Pathkeeper gives an invented reward.",
        sources=_answer().sources,
        refused=False,
        retrieved_chunks=4,
        retrieved_sources=_answer().retrieved_sources,
    )

    record = assess_answer(case, answer, duration_seconds=0.5)

    assert record.passed is False
    assert record.forbidden_keywords_found == ("invented reward",)
    assert record.failure_reason == "forbidden_keyword_present"


def test_missing_topic_passes_only_on_refusal() -> None:
    refused = assess_answer(
        _case(should_answer=False),
        _answer(refused=True),
        duration_seconds=0.5,
    )
    answered = assess_answer(
        _case(should_answer=False),
        _answer(),
        duration_seconds=0.5,
    )

    assert refused.passed is True
    assert answered.passed is False
    assert answered.failure_reason == "bot_answered_missing_topic"


def test_golden_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    line = (
        '{"id":"same","topic":"x","question":"q","should_answer":false,'
        '"expected_source":null,"expected_answer":"unknown","required_keywords":[]}'
    )
    path = tmp_path / "cases.txt"
    _ = path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        _ = load_golden_cases(path)


def test_evaluate_writes_report_and_closes_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case_path = tmp_path / "cases.jsonl"
    result_path = tmp_path / "results.json"
    log_path = tmp_path / "queries.jsonl"
    _ = case_path.write_text(
        json.dumps(
            {
                "id": "case-1",
                "topic": "Pathkeeper",
                "question": "What does Pathkeeper do?",
                "should_answer": True,
                "expected_source": "19-pathkeeper.md",
                "expected_answer": "Pathkeeper explains recipes.",
                "required_keywords": ["Pathkeeper"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeService:
        def __init__(self) -> None:
            self.closed: bool = False

        async def answer(self, question: str) -> RagAnswer:
            assert question == "What does Pathkeeper do?"
            return _answer()

        async def aclose(self) -> None:
            self.closed = True

    service = FakeService()

    def create_fake_service(
        settings: AppSettings,
        *,
        query_log_path: Path,
    ) -> FakeService:
        _ = settings, query_log_path
        return service

    monkeypatch.setattr(
        evaluate_knowledge_base,
        "create_rag_service",
        create_fake_service,
    )

    report = asyncio.run(
        evaluate(
            cases_path=case_path,
            log_path=log_path,
            results_path=result_path,
            settings=AppSettings(repository_root=tmp_path),
        )
    )

    assert report["passed_cases"] == 1
    assert json.loads(result_path.read_text(encoding="utf-8"))["pass_rate"] == 1.0
    assert service.closed is True
