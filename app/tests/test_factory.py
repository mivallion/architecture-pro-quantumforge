from pathlib import Path

import pytest

from app.factory import create_rag_service
from app.settings import AppSettings


def test_factory_fails_early_when_index_is_missing(tmp_path: Path) -> None:
    settings = AppSettings(repository_root=tmp_path)

    with pytest.raises(FileNotFoundError, match="FAISS index not found"):
        _ = create_rag_service(settings)
