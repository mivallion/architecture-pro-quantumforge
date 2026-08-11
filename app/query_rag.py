"""Run one local RAG query without Telegram."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from .factory import create_rag_service


def _parse_question(arguments: Sequence[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="question for the Aetherfall KB")
    namespace = parser.parse_args(arguments)
    return " ".join(namespace.question)


async def _run(question: str) -> None:
    service = create_rag_service()
    try:
        answer = await service.answer(question)
        print(answer.text)
    finally:
        await service.aclose()


def main(arguments: Sequence[str] | None = None) -> None:
    asyncio.run(_run(_parse_question(arguments)))


if __name__ == "__main__":
    main()
