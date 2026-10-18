"""Background task implementation shared by Celery and API fallback."""
from __future__ import annotations
import os
from pathlib import Path
from backend.core import jobs
from backend.services.evidence import fuse_evidence
from backend.services.redaction import redact_text

def process_analysis(job_id: str, data_dir: str="data/raw", uploaded_paths: list[str]|None=None):
    jobs.update(job_id,"running")
    try:
        from risk_engine.findings import run_all
        from document_engine.file_ingestion import ingest_file
        result=run_all(data_dir)
        chunks=[]
        for path in (uploaded_paths or []):
            parsed,_meta=ingest_file(path)
            # Redact snippets before persistence/return; source files remain untouched.
            for chunk in parsed:
                safe=dict(chunk); safe["text"]=redact_text(str(safe.get("text","")))["text"]
                chunks.append(safe)
        result["findings"]=fuse_evidence(result["findings"],chunks)
        serializable={k:v for k,v in result.items() if k not in {"fin","cust","sup","debt","contracts","scored_tx"}}
        # Preserve useful table outputs as JSON-safe records for the API.
        for key in ("fin","cust","sup","debt","contracts","scored_tx"):
            value=result.get(key)
            if value is not None and hasattr(value,"to_dict"):
                serializable[key]=value.to_dict(orient="records")
            elif value is not None:
                serializable[key]=value
        serializable["documents_processed"]=len(uploaded_paths or [])
        jobs.update(job_id,"succeeded",result=serializable)
        return serializable
    except Exception as exc:
        jobs.update(job_id,"failed",error=f"{type(exc).__name__}: {exc}")
        raise

try:
    from backend.workers.celery_app import celery_app
    @celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries":3})
    def analyze_task(self, job_id: str, data_dir: str="data/raw", uploaded_paths=None):
        return process_analysis(job_id,data_dir,uploaded_paths)
except ImportError:
    celery_app=None
    analyze_task=None
