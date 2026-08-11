"""Run the labelled Task 4 dialogs through the complete local RAG pipeline."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

from .factory import create_rag_service
from .settings import AppSettings


class _DialogCase(TypedDict):
    id: str
    question: str
    expected_source: str | None
    should_answer: bool


async def _run() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    cases = cast(
        list[_DialogCase],
        json.loads(
            (repository_root / "Task4" / "eval_cases.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    settings = replace(AppSettings.from_environment(), ollama_keep_alive="5m")
    service = create_rag_service(settings)
    try:
        print("# Проверочные диалоги RAG-бота\n")
        print("Все ответы ниже получены реальным сквозным прогоном retrieval → Ollama.\n")
        for case in cases:
            started = time.perf_counter()
            answer = await service.answer(case["question"])
            duration = time.perf_counter() - started
            expected_kind = "ответ" if case["should_answer"] else "отказ"
            actual_kind = "отказ" if answer.refused else "ответ"
            print(f"## {case['id']}\n")
            print(f"**Ожидание:** {expected_kind}. **Результат:** {actual_kind}. ")
            print(f"**Время:** {duration:.2f} с.\n")
            print(f"**Пользователь:** {case['question']}\n")
            print("**Бот:**\n")
            print(answer.text)
            print()
    finally:
        await service.aclose()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
