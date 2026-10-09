"""Golden-set case model and JSONL validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class GoldenCase:
    case_id: str
    topic: str
    question: str
    should_answer: bool
    expected_source: str | None
    expected_answer: str
    required_keywords: tuple[str, ...]
    forbidden_keywords: tuple[str, ...] = ()


def load_golden_cases(path: Path) -> tuple[GoldenCase, ...]:
    cases: list[GoldenCase] = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        raw = cast(dict[str, object], json.loads(line))
        case = _parse_case(raw, line_number=line_number)
        if case.case_id in identifiers:
            raise ValueError(f"duplicate golden case id: {case.case_id}")
        identifiers.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("golden set must not be empty")
    return tuple(cases)


def _parse_case(raw: dict[str, object], *, line_number: int) -> GoldenCase:
    required = (
        "id",
        "topic",
        "question",
        "should_answer",
        "expected_answer",
        "required_keywords",
    )
    missing_fields = tuple(field for field in required if field not in raw)
    if missing_fields:
        raise ValueError(
            f"golden case line {line_number} misses fields: {', '.join(missing_fields)}"
        )
    case_id = raw["id"]
    topic = raw["topic"]
    question = raw["question"]
    should_answer = raw["should_answer"]
    expected_answer = raw["expected_answer"]
    expected_source = raw.get("expected_source")
    keywords = raw["required_keywords"]
    forbidden_keywords = raw.get("forbidden_keywords", [])
    if not all(
        isinstance(value, str) and value.strip()
        for value in (case_id, topic, question, expected_answer)
    ):
        raise ValueError(f"golden case line {line_number} has an empty text field")
    if not isinstance(should_answer, bool):
        raise ValueError(  # noqa: TRY004 - malformed cases share one validation error API.
            f"golden case line {line_number} has invalid should_answer"
        )
    if expected_source is not None and not isinstance(expected_source, str):
        raise ValueError(f"golden case line {line_number} has invalid expected_source")
    if not isinstance(keywords, list) or not all(
        isinstance(keyword, str) and keyword.strip()
        for keyword in cast(list[object], keywords)
    ):
        raise ValueError(f"golden case line {line_number} has invalid keywords")
    if not isinstance(forbidden_keywords, list) or not all(
        isinstance(keyword, str) and keyword.strip()
        for keyword in cast(list[object], forbidden_keywords)
    ):
        raise ValueError(
            f"golden case line {line_number} has invalid forbidden_keywords"
        )
    if should_answer and expected_source is None:
        raise ValueError(f"known case line {line_number} requires expected_source")
    return GoldenCase(
        case_id=cast(str, case_id),
        topic=cast(str, topic),
        question=cast(str, question),
        should_answer=should_answer,
        expected_source=expected_source,
        expected_answer=cast(str, expected_answer),
        required_keywords=tuple(cast(list[str], keywords)),
        forbidden_keywords=tuple(cast(list[str], forbidden_keywords)),
    )
