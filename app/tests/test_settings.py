from pathlib import Path

from app.settings import AppSettings


def test_settings_load_environment_overrides() -> None:
    settings = AppSettings.from_environment(
        {
            "RAG_REPOSITORY_ROOT": "/tmp/project",
            "OLLAMA_MODEL": "local-model",
            "EMBEDDING_LOCAL_FILES_ONLY": "false",
            "RAG_TOP_K": "3",
            "RAG_SCORE_THRESHOLD": "0.42",
        }
    )

    assert settings.repository_root == Path("/tmp/project")
    assert settings.ollama_model == "local-model"
    assert settings.embedding_local_files_only is False
    assert settings.top_k == 3
    assert settings.score_threshold == 0.42
    assert settings.faiss_path == Path("/tmp/project/Task3/index/faiss.index")


def test_blank_threshold_disables_retrieval_gate() -> None:
    settings = AppSettings.from_environment({"RAG_SCORE_THRESHOLD": ""})

    assert settings.score_threshold is None
    assert settings.embedding_local_files_only is True
