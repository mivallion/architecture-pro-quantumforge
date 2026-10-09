from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest
from telegram import Update
from telegram.ext import ContextTypes

from app.telegram_bot import (
    MAX_REPLY_CHARS,
    SEMAPHORE_KEY,
    SERVICE_KEY,
    answer_message,
    load_settings,
    split_telegram_text,
    unsupported_command,
    unsupported_message,
)


@dataclass(frozen=True, slots=True)
class FakeAnswer:
    text: str


class FakeService:
    def __init__(self, answer: str = "Ответ из базы") -> None:
        self.answer_text = answer
        self.questions: list[str] = []
        self.closed = False

    async def answer(self, question: str) -> FakeAnswer:
        self.questions.append(question)
        return FakeAnswer(self.answer_text)

    async def aclose(self) -> None:
        self.closed = True


class FakeMessage:
    def __init__(self, text: str | None) -> None:
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


class FakeChat:
    def __init__(self) -> None:
        self.actions: list[object] = []

    async def send_action(self, action: object) -> None:
        self.actions.append(action)


def _context(service: FakeService, *, max_question_chars: int = 2_000) -> ContextTypes.DEFAULT_TYPE:
    application = SimpleNamespace(
        bot_data={
            SERVICE_KEY: service,
            SEMAPHORE_KEY: asyncio.Semaphore(1),
            "max_question_chars": max_question_chars,
            "max_reply_chars": MAX_REPLY_CHARS,
        }
    )
    return cast(ContextTypes.DEFAULT_TYPE, SimpleNamespace(application=application))


def _update(message: FakeMessage, chat: FakeChat | None = None) -> Update:
    return cast(Update, SimpleNamespace(effective_message=message, effective_chat=chat))


def test_answer_message_queries_service_and_returns_reply() -> None:
    service = FakeService("Найденный ответ")
    message = FakeMessage("  Кто лечит раненых?  ")
    chat = FakeChat()

    asyncio.run(answer_message(_update(message, chat), _context(service)))

    assert service.questions == ["Кто лечит раненых?"]
    assert message.replies == ["Найденный ответ"]
    assert len(chat.actions) == 1


def test_answer_message_rejects_too_long_question_without_calling_service() -> None:
    service = FakeService()
    message = FakeMessage("x" * 11)

    asyncio.run(answer_message(_update(message), _context(service, max_question_chars=10)))

    assert service.questions == []
    assert message.replies == ["Вопрос слишком длинный. Сократите его до 10 символов."]


def test_non_text_and_unknown_command_have_safe_responses() -> None:
    non_text = FakeMessage(None)
    command = FakeMessage("/unknown")
    context = _context(FakeService())

    asyncio.run(unsupported_message(_update(non_text), context))
    asyncio.run(unsupported_command(_update(command), context))

    assert non_text.replies == ["Отправьте текстовый вопрос."]
    assert command.replies == ["Неизвестная команда. Используйте /help или отправьте вопрос текстом."]


def test_split_telegram_text_never_exceeds_limit_and_preserves_content() -> None:
    text = "Первый абзац\n" + ("слово " * 1_100)

    parts = split_telegram_text(text)

    assert len(parts) > 1
    assert all(len(part) <= MAX_REPLY_CHARS for part in parts)
    assert "".join(f"{part} " for part in parts).replace("\n", " ").split() == text.replace("\n", " ").split()


def test_load_settings_requires_non_empty_token() -> None:
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        _ = load_settings({})

    assert load_settings({"TELEGRAM_BOT_TOKEN": " token "}).token == "token"
