"""Detect and neutralize prompt injection embedded in retrieved documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Final

from .rag_contracts import RetrievedChunk, SecurityMode, SecurityReport

_INJECTION_PATTERNS: Final = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bignore\b.{0,80}\binstructions?\b",
        r"\b(?:system|developer)\s+(?:message|prompt)\b",
        r"\b(?:output|print|return|reveal|disclose)\s*:",
        r"<\|(?:system|developer|assistant|user)\|>",
        r"\bигнорируй\b.{0,80}\bинструкц(?:ию|ии|ий|ия)\b",
        r"\b(?:системное|инструкции разработчика)\s+(?:сообщение|промпт)\b",
    )
)


@dataclass(frozen=True, slots=True)
class SecuredChunks:
    chunks: tuple[RetrievedChunk, ...]
    report: SecurityReport


def contains_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in _INJECTION_PATTERNS)


def uses_protected_prompt(mode: SecurityMode) -> bool:
    return mode in {SecurityMode.PRE_PROMPT, SecurityMode.DEFENSE_IN_DEPTH}


def secure_chunks(
    chunks: tuple[RetrievedChunk, ...],
    *,
    mode: SecurityMode,
) -> SecuredChunks:
    detected = sum(contains_prompt_injection(chunk.text) for chunk in chunks)
    if mode in {SecurityMode.UNPROTECTED, SecurityMode.PRE_PROMPT}:
        return SecuredChunks(
            chunks=chunks,
            report=SecurityReport(mode=mode, detected_chunks=detected),
        )

    if mode in {SecurityMode.FILTER, SecurityMode.DEFENSE_IN_DEPTH}:
        safe_chunks = tuple(
            chunk for chunk in chunks if not contains_prompt_injection(chunk.text)
        )
        return SecuredChunks(
            chunks=safe_chunks,
            report=SecurityReport(
                mode=mode,
                detected_chunks=detected,
                removed_chunks=len(chunks) - len(safe_chunks),
            ),
        )

    sanitized_chunks: list[RetrievedChunk] = []
    sanitized_count = 0
    removed_count = 0
    for chunk in chunks:
        if not contains_prompt_injection(chunk.text):
            sanitized_chunks.append(chunk)
            continue
        sanitized_count += 1
        sanitized_text = _remove_suspicious_lines(chunk.text)
        if not sanitized_text:
            removed_count += 1
            continue
        sanitized_chunks.append(replace(chunk, text=sanitized_text))
    return SecuredChunks(
        chunks=tuple(sanitized_chunks),
        report=SecurityReport(
            mode=mode,
            detected_chunks=detected,
            removed_chunks=removed_count,
            sanitized_chunks=sanitized_count,
        ),
    )


def _remove_suspicious_lines(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not contains_prompt_injection(line)
    ).strip()
