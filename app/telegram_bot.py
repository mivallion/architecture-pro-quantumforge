"""Thin Telegram transport adapter for the local RAG service.

The adapter deliberately owns no retrieval or LLM resources.  A composition
root supplies a long-lived service, so handlers remain inexpensive and easily
testable without a network connection.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

LOGGER = logging.getLogger(__name__)
SERVICE_KEY = "rag_service"
SEMAPHORE_KEY = "rag_semaphore"
MAX_QUESTION_CHARS = 2_000
MAX_REPLY_CHARS = 4_000


class RagAnswer(Protocol):
    """The response shape required by the Telegram transport."""

    @property
    def text(self) -> str: ...


class RagService(Protocol):
    """Structural interface to avoid importing the RAG composition module."""

    async def answer(self, question: str) -> RagAnswer: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BotSettings:
    token: str
    max_question_chars: int = MAX_QUESTION_CHARS
    max_reply_chars: int = MAX_REPLY_CHARS


def load_settings(environment: Mapping[str, str] | None = None) -> BotSettings:
    """Read the bot token without ever logging or persisting it."""
    source = os.environ if environment is None else environment
    token = source.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is required")
    return BotSettings(token=token)


def build_application(settings: BotSettings, rag_service: RagService) -> Application:
    """Build a sequential polling application around an existing RAG service."""
    application = (
        Application.builder()
        .token(settings.token)
        .concurrent_updates(1)
        .post_shutdown(_post_shutdown)
        .build()
    )
    application.bot_data[SERVICE_KEY] = rag_service
    application.bot_data[SEMAPHORE_KEY] = asyncio.Semaphore(1)
    application.bot_data["max_question_chars"] = settings.max_question_chars
    application.bot_data["max_reply_chars"] = settings.max_reply_chars
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.COMMAND, unsupported_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer_message))
    application.add_handler(MessageHandler(~filters.TEXT, unsupported_message))
    application.add_error_handler(error_handler)
    return application


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _ = context
    await _reply(update, "Я отвечаю только по базе знаний. Отправьте текстовый вопрос.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _ = context
    await _reply(
        update,
        "Например: «Кто лечит раненых поселенцев?»\n"
        "Если в базе нет ответа, я честно скажу: «Я не знаю».",
    )


async def unsupported_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _ = context
    await _reply(update, "Неизвестная команда. Используйте /help или отправьте вопрос текстом.")


async def unsupported_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _ = context
    await _reply(update, "Отправьте текстовый вопрос.")


async def answer_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Validate a question, show typing, and serialize resource-intensive RAG calls."""
    message = update.effective_message
    if message is None or message.text is None:
        return
    question = message.text.strip()
    if not question:
        return

    maximum = _bot_data_int(context, "max_question_chars", MAX_QUESTION_CHARS)
    if len(question) > maximum:
        await message.reply_text(f"Вопрос слишком длинный. Сократите его до {maximum} символов.")
        return

    semaphore = _bot_data_semaphore(context)
    service = _bot_data_service(context)
    async with semaphore:
        if update.effective_chat is not None:
            await update.effective_chat.send_action(ChatAction.TYPING)
        answer = await service.answer(question)

    maximum_reply = _bot_data_int(context, "max_reply_chars", MAX_REPLY_CHARS)
    for part in split_telegram_text(answer.text, maximum_reply):
        await message.reply_text(part)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Keep diagnostics in logs while never exposing internals to a chat."""
    LOGGER.error("Unhandled error while processing a Telegram update", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message is not None:
        message = cast(Update, update).effective_message
        if message is None:
            return
        await message.reply_text(
            "Не удалось обработать запрос. Попробуйте ещё раз позже."
        )


async def _post_shutdown(application: Application) -> None:
    service = application.bot_data.get(SERVICE_KEY)
    if service is not None:
        await _as_rag_service(service).aclose()


def split_telegram_text(text: str, maximum: int = MAX_REPLY_CHARS) -> tuple[str, ...]:
    """Split plain text at paragraph/word boundaries without exceeding a limit."""
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if not text:
        return ("Я не знаю: в базе знаний недостаточно информации.",)

    parts: list[str] = []
    remaining = text
    while len(remaining) > maximum:
        split_at = max(remaining.rfind("\n", 0, maximum + 1), remaining.rfind(" ", 0, maximum + 1))
        if split_at <= 0:
            split_at = maximum
        parts.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        parts.append(remaining)
    return tuple(parts)


def _bot_data_service(context: ContextTypes.DEFAULT_TYPE) -> RagService:
    return _as_rag_service(context.application.bot_data[SERVICE_KEY])


def _as_rag_service(value: object) -> RagService:
    if not hasattr(value, "answer") or not hasattr(value, "aclose"):
        raise RuntimeError("RAG service is not configured")
    return cast(RagService, value)


def _bot_data_semaphore(context: ContextTypes.DEFAULT_TYPE) -> asyncio.Semaphore:
    value = context.application.bot_data[SEMAPHORE_KEY]
    if not isinstance(value, asyncio.Semaphore):
        raise TypeError("RAG semaphore is not configured")
    return value


def _bot_data_int(context: ContextTypes.DEFAULT_TYPE, key: str, default: int) -> int:
    value = context.application.bot_data.get(key, default)
    return value if isinstance(value, int) and value > 0 else default


async def _reply(update: Update, text: str) -> None:
    if update.effective_message is not None:
        await update.effective_message.reply_text(text)
