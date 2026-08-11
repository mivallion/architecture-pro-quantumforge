"""Validate the configured Telegram token without printing secret values."""

from __future__ import annotations

import asyncio
from pathlib import Path

from telegram import Bot
from telegram.error import InvalidToken, TelegramError

from .bot import load_repository_environment
from .telegram_bot import load_settings


async def _check() -> int:
    repository_root = Path(__file__).resolve().parent.parent
    _ = load_repository_environment(repository_root)
    try:
        settings = load_settings()
        async with Bot(settings.token) as bot:
            identity = await bot.get_me()
    except InvalidToken:
        print("Telegram отклонил токен.")
        return 2
    except TelegramError as error:
        print(
            "Не удалось проверить токен из-за ошибки Telegram или сети "
            f"({type(error).__name__})."
        )
        return 3
    except Exception as error:
        print(f"Не удалось проверить конфигурацию ({type(error).__name__}).")
        return 4

    username = identity.username or "без username"
    print(f"Токен принят Telegram. Бот: @{username}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_check()))


if __name__ == "__main__":
    main()
