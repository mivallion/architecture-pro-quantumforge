from __future__ import annotations

import os
from pathlib import Path

from pytest import MonkeyPatch

from app.bot import load_repository_environment


def test_load_repository_environment_reads_env_without_overriding(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "TELEGRAM_BOT_TOKEN=from-file\nRAG_TOP_K=3\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "from-process")
    monkeypatch.delenv("RAG_TOP_K", raising=False)

    loaded = load_repository_environment(tmp_path)

    assert loaded is True
    assert os.environ["TELEGRAM_BOT_TOKEN"] == "from-process"
    assert os.environ["RAG_TOP_K"] == "3"
