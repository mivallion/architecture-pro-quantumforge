"""Durable structured event logging for scheduled index updates."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class JsonlEventWriter:
    def __init__(self, path: Path) -> None:
        self._path: Path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: JsonValue) -> None:
        record: dict[str, JsonValue] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._path.open("a", encoding="utf-8") as stream:
            _ = stream.write(serialized + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(serialized, flush=True)


__all__ = ["JsonlEventWriter"]
