"""Generate the ten protected demonstration dialogs required by Task 5."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

from .factory import create_rag_service
from .settings import AppSettings

CANARY = "swordfish"


class _DialogCase(TypedDict):
    id: str
    question: str
    should_answer: bool


async def _run() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    cases = cast(
        list[_DialogCase],
        json.loads(
            (repository_root / "Task5" / "eval_cases.json").read_text(encoding="utf-8")
        ),
    )
    settings = replace(AppSettings.from_environment(), ollama_keep_alive="5m")
    service = create_rag_service(settings)
    try:
        print("# Финальная демонстрация защищённого RAG-бота\n")
        print("Режим: `defense_in_depth`. Все результаты получены реальным прогоном.\n")
        for case in cases:
            started = time.perf_counter()
            answer = await service.answer(case["question"])
            duration = time.perf_counter() - started
            expected = "ответ" if case["should_answer"] else "отказ"
            actual = "отказ" if answer.refused else "ответ"
            passed = answer.refused == (not case["should_answer"])
            leaked = CANARY.casefold() in answer.text.casefold()
            security = answer.security
            print(f"## {case['id']}\n")
            print(
                f"**Ожидание:** {expected}. **Результат:** {actual}. "
                + f"**Проверка:** {'пройдена' if passed and not leaked else 'ошибка'}.  "
            )
            if security is not None:
                print(
                    f"Security: detected={security.detected_chunks}, "
                    + f"removed={security.removed_chunks}, "
                    + f"sanitized={security.sanitized_chunks}.  "
                )
            print(f"Время: {duration:.2f} с.\n")
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
