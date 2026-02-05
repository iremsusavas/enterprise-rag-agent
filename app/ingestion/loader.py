from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


def load_documents(file_path: str) -> List[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    loader = _select_loader(path)
    return loader.load()


def _select_loader(path: Path):
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(str(path))

    if suffix in {".md", ".markdown"}:
        return UnstructuredMarkdownLoader(str(path))

    return TextLoader(str(path), autodetect_encoding=True)
