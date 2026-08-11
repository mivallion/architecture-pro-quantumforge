"""Run the Telegram interface with ``python -m app.bot``."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from telegram import Update

from .factory import create_rag_service
from .settings import AppSettings
from .telegram_bot import BotSettings, build_application, load_settings


def load_repository_environment(repository_root: Path) -> bool:
    """Load local development secrets without overriding explicit environment."""
    return load_dotenv(
        dotenv_path=repository_root / ".env",
        override=False,
        interpolate=False,
    )


def main() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    _ = load_repository_environment(repository_root)
    app_settings = AppSettings.from_environment()
    token_settings = load_settings()
    service = create_rag_service(app_settings)
    application = build_application(
        BotSettings(
            token=token_settings.token,
            max_question_chars=app_settings.max_question_chars,
        ),
        service,
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
