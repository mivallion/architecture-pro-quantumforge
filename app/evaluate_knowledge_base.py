"""Run the Task 7 golden set through the complete local RAG service."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

from .evaluation_cases import GoldenCase, load_golden_cases
from .evaluation_scoring import EvaluationRecord, assess_answer
from .factory import create_rag_service
from .settings import AppSettings


async def evaluate(
    *,
    cases_path: Path,
    log_path: Path,
    results_path: Path,
    settings: AppSettings | None = None,
    reset_log: bool = False,
) -> dict[str, object]:
    cases = load_golden_cases(cases_path)
    if reset_log:
        log_path.unlink(missing_ok=True)
    resolved_settings = settings or replace(
        AppSettings.from_environment(),
        ollama_keep_alive="5m",
    )
    service = create_rag_service(resolved_settings, query_log_path=log_path)
    records: list[EvaluationRecord] = []
    try:
        for case in cases:
            started = time.perf_counter()
            try:
                answer = await service.answer(case.question)
            except Exception as error:  # noqa: BLE001 - each external failure becomes a report record.
                records.append(
                    EvaluationRecord(
                        case_id=case.case_id,
                        topic=case.topic,
                        should_answer=case.should_answer,
                        actual_status="error",
                        passed=False,
                        source_match=None,
                        keywords_match=None,
                        forbidden_keywords_found=(),
                        answer_length=0,
                        retrieved_chunks=0,
                        retrieved_sources=(),
                        sources=(),
                        retrieval_source_match=None,
                        duration_seconds=round(time.perf_counter() - started, 3),
                        failure_reason=type(error).__name__,
                        answer="",
                    )
                )
                continue
            records.append(
                assess_answer(
                    case,
                    answer,
                    duration_seconds=time.perf_counter() - started,
                )
            )
    finally:
        await service.aclose()

    known = tuple(record for record in records if record.should_answer)
    missing = tuple(record for record in records if not record.should_answer)
    report: dict[str, object] = {
        "generated_at": _latest_log_timestamp(log_path),
        "total_cases": len(records),
        "passed_cases": sum(record.passed for record in records),
        "pass_rate": _ratio(sum(record.passed for record in records), len(records)),
        "known_cases": len(known),
        "known_passed": sum(record.passed for record in known),
        "known_pass_rate": _ratio(sum(record.passed for record in known), len(known)),
        "missing_cases": len(missing),
        "missing_passed": sum(record.passed for record in missing),
        "missing_refusal_rate": _ratio(
            sum(record.actual_status == "refused" for record in missing),
            len(missing),
        ),
        "detected_gaps": [
            record.topic for record in missing if record.actual_status == "refused"
        ],
        "failures": [record.case_id for record in records if not record.passed],
        "cases": [asdict(record) for record in records],
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    _ = results_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 4)


def _latest_log_timestamp(path: Path) -> str | None:
    if not path.is_file():
        return None
    lines = tuple(
        line for line in path.read_text(encoding="utf-8").splitlines() if line
    )
    if not lines:
        return None
    value = cast(dict[str, object], json.loads(lines[-1])).get("timestamp")
    return value if isinstance(value, str) else None


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--cases",
        type=Path,
        default=repository_root / "Task7" / "golden_questions.txt",
    )
    _ = parser.add_argument(
        "--log-file",
        type=Path,
        default=repository_root / "Task7" / "logs.jsonl",
    )
    _ = parser.add_argument(
        "--results",
        type=Path,
        default=repository_root / "Task7" / "evaluation_results.json",
    )
    _ = parser.add_argument("--reset-log", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    namespace = _parse_arguments(arguments)
    report = asyncio.run(
        evaluate(
            cases_path=namespace.cases,
            log_path=namespace.log_file,
            results_path=namespace.results,
            reset_log=namespace.reset_log,
        )
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "EvaluationRecord",
    "GoldenCase",
    "assess_answer",
    "evaluate",
    "load_golden_cases",
]
