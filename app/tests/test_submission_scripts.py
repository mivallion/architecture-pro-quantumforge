from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from app import build_index as build_index_script
from app import evaluate_prompt_injection, generate_task5_logs
from app.index_contracts import BuildResult
from app.indexing import RecursiveChunker
from app.rag_contracts import ChatMessage, RagAnswer, RetrievedChunk
from app.settings import AppSettings


def test_build_index_cli_reports_build_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = BuildResult(
        scanned_documents=3,
        indexed_documents=1,
        indexed_chunks=2,
        unchanged_documents=2,
        deleted_documents=0,
        total_chunks=4,
        index_changed=True,
        duration_seconds=0.25,
    )
    embedder = SimpleNamespace(tokenizer=object())

    def create_chunker(
        tokenizer: object,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> object:
        _ = tokenizer, chunk_size, chunk_overlap
        return object()

    def return_result(*args: object) -> BuildResult:
        _ = args
        return result

    monkeypatch.setattr(build_index_script, "QwenEmbedder", lambda: embedder)
    monkeypatch.setattr(
        RecursiveChunker,
        "from_huggingface_tokenizer",
        create_chunker,
    )
    monkeypatch.setattr(build_index_script, "build_index", return_result)

    build_index_script.main()

    output = capsys.readouterr().out
    assert "Indexed 1/3 documents" in output
    assert "into 4 total chunks in 0.25s" in output
    assert "changed=True" in output


def test_prompt_injection_experiment_runs_every_mode_and_closes_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chunk = RetrievedChunk(
        chunk_id=1,
        score=0.9,
        title="Safe",
        source_path="safe.md",
        chunk_number=0,
        start_index=0,
        text="Safe reference text.",
    )

    class FakeRetriever:
        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs

        def retrieve(self, query: str, *, limit: int) -> tuple[RetrievedChunk, ...]:
            assert query == evaluate_prompt_injection.QUESTION
            assert limit == 4
            return (chunk,)

    class FakeModel:
        instances: ClassVar[list[FakeModel]] = []

        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs
            self.closed: bool = False
            self.instances.append(self)

        async def generate(self, messages: tuple[ChatMessage, ...]) -> str:
            _ = messages
            return "Safe answer"

        async def aclose(self) -> None:
            self.closed = True

    settings = AppSettings(repository_root=tmp_path)

    def load_settings(cls: type[AppSettings]) -> AppSettings:
        _ = cls
        return settings

    def create_embedder(**kwargs: object) -> object:
        _ = kwargs
        return object()

    monkeypatch.setattr(
        AppSettings,
        "from_environment",
        classmethod(load_settings),
    )
    monkeypatch.setattr(evaluate_prompt_injection, "QwenEmbedder", create_embedder)
    monkeypatch.setattr(evaluate_prompt_injection, "FaissRetriever", FakeRetriever)
    monkeypatch.setattr(evaluate_prompt_injection, "OllamaChatModel", FakeModel)

    evaluate_prompt_injection.main()

    output = capsys.readouterr().out
    assert output.count("canary попала в ответ: нет") == len(
        evaluate_prompt_injection.MODES
    )
    assert all(f"## {mode.value}" in output for mode in evaluate_prompt_injection.MODES)
    assert FakeModel.instances[0].closed is True


def test_dialog_generator_processes_all_cases_and_closes_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeService:
        def __init__(self) -> None:
            self.questions: list[str] = []
            self.closed: bool = False

        async def answer(self, question: str) -> RagAnswer:
            self.questions.append(question)
            return RagAnswer(text="Safe answer", sources=(), refused=False)

        async def aclose(self) -> None:
            self.closed = True

    service = FakeService()
    settings = AppSettings(repository_root=tmp_path)

    def load_settings(cls: type[AppSettings]) -> AppSettings:
        _ = cls
        return settings

    def create_service(resolved_settings: AppSettings) -> FakeService:
        _ = resolved_settings
        return service

    monkeypatch.setattr(
        AppSettings,
        "from_environment",
        classmethod(load_settings),
    )
    monkeypatch.setattr(generate_task5_logs, "create_rag_service", create_service)

    generate_task5_logs.main()

    output = capsys.readouterr().out
    assert len(service.questions) == 10
    assert output.count("## ") == 10
    assert service.closed is True
