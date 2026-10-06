"""Shared DealLens pipeline and evidence registry."""
from __future__ import annotations
from pathlib import Path
import shutil

from risk_engine.findings import run_all
from document_engine.file_ingestion import ingest_file
from document_engine.search import DocIndex

STATE: dict = {}
DOCS: dict = {"chunks": [], "index": None, "files": []}


def analyze(data_dir: str = "data/raw") -> dict:
    res = run_all(data_dir)
    STATE.clear()
    STATE.update(res)
    return res


def reset_documents() -> None:
    DOCS["chunks"] = []
    DOCS["index"] = None
    DOCS["files"] = []


def add_file(path: str) -> dict:
    chunks, meta = ingest_file(path)
    DOCS["chunks"].extend(chunks)
    DOCS["index"] = DocIndex(DOCS["chunks"])
    if meta["source"] not in DOCS["files"]:
        DOCS["files"].append(meta["source"])
    return {**meta, "chunks_indexed": len(chunks), "total_chunks": len(DOCS["chunks"])}


def add_pdf(path: str) -> int:
    return int(add_file(path)["chunks_indexed"])


def save_upload(uploaded_file, directory: str = "data/uploads") -> str:
    Path(directory).mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    destination = Path(directory) / safe_name
    with destination.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(destination)
