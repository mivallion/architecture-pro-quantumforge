"""Pure scoring for golden-set RAG responses."""

from __future__ import annotations

from dataclasses import dataclass

from .evaluation_cases import GoldenCase
from .rag_contracts import RagAnswer


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    case_id: str
    topic: str
    should_answer: bool
    actual_status: str
    passed: bool
    source_match: bool | None
    keywords_match: bool | None
    forbidden_keywords_found: tuple[str, ...]
    answer_length: int
    retrieved_chunks: int
    retrieved_sources: tuple[str, ...]
    sources: tuple[str, ...]
    retrieval_source_match: bool | None
    duration_seconds: float
    failure_reason: str | None
    answer: str


def assess_answer(
    case: GoldenCase,
    answer: RagAnswer,
    *,
    duration_seconds: float,
) -> EvaluationRecord:
    source_paths = tuple(dict.fromkeys(source.source_path for source in answer.sources))
    retrieved_source_paths = tuple(
        dict.fromkeys(source.source_path for source in answer.retrieved_sources)
    )
    normalized_answer = answer.text.casefold()
    source_match = (
        None if case.expected_source is None else case.expected_source in source_paths
    )
    retrieval_source_match = (
        None
        if case.expected_source is None
        else case.expected_source in retrieved_source_paths
    )
    keywords_match = (
        None
        if not case.required_keywords
        else all(
            keyword.casefold() in normalized_answer
            for keyword in case.required_keywords
        )
    )
    forbidden_keywords_found = tuple(
        keyword
        for keyword in case.forbidden_keywords
        if keyword.casefold() in normalized_answer
    )

    if case.should_answer:
        passed = bool(
            not answer.refused
            and source_match
            and keywords_match is not False
            and not forbidden_keywords_found
        )
        if answer.refused:
            failure_reason = "bot_refused_known_question"
        elif source_match is False:
            failure_reason = "expected_source_not_cited"
        elif forbidden_keywords_found:
            failure_reason = "forbidden_keyword_present"
        elif keywords_match is False:
            failure_reason = "required_keyword_missing"
        else:
            failure_reason = None
    else:
        passed = answer.refused
        failure_reason = None if passed else "bot_answered_missing_topic"

    return EvaluationRecord(
        case_id=case.case_id,
        topic=case.topic,
        should_answer=case.should_answer,
        actual_status="refused" if answer.refused else "answered",
        passed=passed,
        source_match=source_match,
        keywords_match=keywords_match,
        forbidden_keywords_found=forbidden_keywords_found,
        answer_length=len(answer.text),
        retrieved_chunks=answer.retrieved_chunks,
        retrieved_sources=retrieved_source_paths,
        sources=source_paths,
        retrieval_source_match=retrieval_source_match,
        duration_seconds=round(duration_seconds, 3),
        failure_reason=failure_reason,
        answer=answer.text,
    )
