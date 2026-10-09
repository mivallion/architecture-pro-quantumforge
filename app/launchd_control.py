"""Validate and restart the local launchd-managed Telegram bot."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Final

_LAUNCHD_LABEL_PATTERN: Final = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_launchd_label(label: str) -> None:
    if _LAUNCHD_LABEL_PATTERN.fullmatch(label) is None:
        raise ValueError("invalid launchd label")


def restart_launchd_bot(label: str) -> None:
    validate_launchd_label(label)
    _ = subprocess.run(
        (
            "/bin/launchctl",
            "kickstart",
            "-k",
            f"gui/{os.getuid()}/{label}",
        ),
        check=True,
        capture_output=True,
        text=True,
    )


__all__ = ["restart_launchd_bot", "validate_launchd_label"]
