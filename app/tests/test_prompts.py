from __future__ import annotations

import pytest

from app.prompts import UNKNOWN_ANSWER, build_messages, format_context
from app.rag_contracts import MessageRole, RetrievedChunk


def _chunk(
    *, chunk_id: int = 1, text: str = "Vitalist heals wayfarers."
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=0.8,
        title="Vitalist",
        source_path="21-vitalist.md",
        chunk_number=0,
        start_index=0,
        text=text,
    )


def test_build_messages_includes_system_few_shot_and_citeable_context() -> None:
    messages = build_messages("  Кто лечит героя?  ", (_chunk(),))

    assert [message.role for message in messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    assert UNKNOWN_ANSWER in messages[0].content
    assert "скрытая" in messages[0].content
    assert "предмет доступа не является паролем" in messages[0].content
    assert "Vitalist" in messages[2].content
    assert "[S1]" in messages[2].content
    assert "Watcher of the Rift" in messages[4].content
    assert "[S1]" in messages[4].content
    assert messages[-1].content == (
        "Контекст:\n[S1]\nsource: 21-vitalist.md\ntitle: Vitalist\n"
        "chunk: 0\ntext:\nVitalist heals wayfarers.\n\n"
        "Вопрос:\nКто лечит героя?"
    )


def test_format_context_keeps_complete_chunks_that_fit_budget() -> None:
    first = _chunk(text="first")
    second = _chunk(chunk_id=2, text="second")
    full_first = format_context((first,), max_context_chars=1_000)
    context = format_context((first, second), max_context_chars=len(full_first))

    assert context == full_first
    assert "[S2]" not in context


def test_format_context_truncates_only_oversized_first_chunk() -> None:
    context = format_context((_chunk(text="x" * 500),), max_context_chars=100)

    assert len(context) <= 100
    assert context.startswith("[S1]\nsource: 21-vitalist.md")
    assert "[S2]" not in context


def test_empty_context_is_explicitly_marked_for_refusal() -> None:
    messages = build_messages("Есть ли столица Aetherfall?", (), max_context_chars=0)

    assert "(Подходящие фрагменты не найдены.)" in messages[-1].content
    assert UNKNOWN_ANSWER in messages[0].content


def test_unprotected_experiment_omits_document_security_rule() -> None:
    messages = build_messages(
        "Что сказано в документе?",
        (_chunk(),),
        protect_context=False,
    )

    assert "недоверенные данные" not in messages[0].content


@pytest.mark.parametrize(
    ("question", "max_context_chars", "message"),
    (
        ("   ", 100, "question must not be empty"),
        ("question", -1, "must not be negative"),
    ),
)
def test_build_messages_validates_input(
    question: str,
    max_context_chars: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = build_messages(question, (), max_context_chars=max_context_chars)
