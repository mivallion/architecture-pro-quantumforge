from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NewType, override

SourceTitle = NewType("SourceTitle", str)
FictionalName = NewType("FictionalName", str)


class EntityKind(StrEnum):
    SOVEREIGN = "sovereign"
    SETTLER = "settler"


@dataclass(frozen=True, slots=True)
class Term:
    source: SourceTitle
    replacement: FictionalName


@dataclass(frozen=True, slots=True)
class EntitySpec:
    source_title: SourceTitle
    fictional_name: FictionalName
    kind: EntityKind


@dataclass(frozen=True, slots=True)
class Document:
    name: FictionalName
    kind: EntityKind
    body: str


@dataclass(frozen=True, slots=True)
class WikiPage:
    title: SourceTitle
    raw: str


@dataclass(frozen=True, slots=True)
class MissingPagesError(Exception):
    titles: tuple[SourceTitle, ...]

    @override
    def __str__(self) -> str:
        return f"missing pages: {', '.join(self.titles)}"


@dataclass(frozen=True, slots=True)
class ValidationError(Exception):
    issues: tuple[str, ...]

    @override
    def __str__(self) -> str:
        return "knowledge base validation failed:\n- " + "\n- ".join(self.issues)


@dataclass(frozen=True, slots=True)
class TermsMapError(Exception):
    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        return f"cannot load {self.path}: {self.reason}"


@dataclass(frozen=True, slots=True)
class MissingEntityTermsError(Exception):
    titles: tuple[SourceTitle, ...]

    @override
    def __str__(self) -> str:
        return f"terms map has no replacements for: {', '.join(self.titles)}"
