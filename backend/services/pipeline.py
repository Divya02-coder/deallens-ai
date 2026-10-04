"""Shared pipeline used by API + UI. Holds in-memory state (swap for PostgreSQL later)."""
from pathlib import Path
from risk_engine.findings import run_all
from document_engine.pdf_parser import parse_pdf, chunk_pages
from document_engine.search import DocIndex

STATE: dict = {}
DOCS: dict = {"chunks": [], "index": None}


def analyze(data_dir: str = "data/raw") -> dict:
    res = run_all(data_dir)
    STATE.clear(); STATE.update(res)
    return res


def add_pdf(path: str) -> int:
    chunks = chunk_pages(parse_pdf(path))
    DOCS["chunks"] += chunks
    DOCS["index"] = DocIndex(DOCS["chunks"])
    return len(chunks)
