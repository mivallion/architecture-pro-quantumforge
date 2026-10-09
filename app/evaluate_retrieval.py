"""Evaluate Task 3 retrieval against the labelled Task 4 questions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict, cast

from .embeddings import QwenEmbedder
from .retrieval import FaissRetriever
from .settings import AppSettings


class _RawCase(TypedDict):
    id: str
    question: str
    expected_source: str | None
    should_answer: bool


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    case_id: str
    should_answer: bool
    expected_source: str | None
    top_score: float | None
    retrieved_sources: tuple[str, ...]
    expected_in_top_k: bool | None


def evaluate(*, cases_path: Path, top_k: int = 4) -> dict[str, object]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    raw_cases = cast(list[_RawCase], json.loads(cases_path.read_text(encoding="utf-8")))
    settings = AppSettings.from_environment()
    retriever = FaissRetriever(
        QwenEmbedder(local_files_only=settings.embedding_local_files_only),
        faiss_path=settings.faiss_path,
        database_path=settings.database_path,
    )

    results: list[EvaluationResult] = []
    for case in raw_cases:
        chunks = retriever.retrieve(case["question"], limit=top_k)
        sources = tuple(Path(chunk.source_path).name for chunk in chunks)
        expected = case["expected_source"]
        results.append(
            EvaluationResult(
                case_id=case["id"],
                should_answer=case["should_answer"],
                expected_source=expected,
                top_score=chunks[0].score if chunks else None,
                retrieved_sources=sources,
                expected_in_top_k=None if expected is None else expected in sources,
            )
        )

    known = [result for result in results if result.should_answer]
    unknown = [result for result in results if not result.should_answer]
    known_scores = [result.top_score for result in known if result.top_score is not None]
    unknown_scores = [result.top_score for result in unknown if result.top_score is not None]
    minimum_known = min(known_scores, default=None)
    maximum_unknown = max(unknown_scores, default=None)
    suggested_threshold = (
        (minimum_known + maximum_unknown) / 2
        if minimum_known is not None
        and maximum_unknown is not None
        and maximum_unknown < minimum_known
        else None
    )
    known_hits = sum(result.expected_in_top_k is True for result in known)
    return {
        "top_k": top_k,
        "known_source_recall_at_k": known_hits / len(known) if known else None,
        "minimum_known_top_score": minimum_known,
        "maximum_unknown_top_score": maximum_unknown,
        "suggested_score_threshold": suggested_threshold,
        "note": (
            "Score gate is separable on this sample."
            if suggested_threshold is not None
            else "No reliable score-only gate; keep grounded LLM refusal and citation checks."
        ),
        "cases": [asdict(result) for result in results],
    }


def main(arguments: Sequence[str] | None = None) -> None:
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=repository_root / "Task4" / "eval_cases.json",
    )
    parser.add_argument("--top-k", type=int, default=4)
    namespace = parser.parse_args(arguments)
    report = evaluate(cases_path=namespace.cases, top_k=namespace.top_k)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
