"""Versioned document and analysis API."""
from __future__ import annotations
import os, uuid
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from backend.core import jobs
from backend.services.redaction import redact_text
from backend.services.model_provider import generate_local

router=APIRouter(prefix="/api/v1")
UPLOAD_DIR=Path(os.getenv("DEALLENS_UPLOAD_DIR","data/uploads"))
ALLOWED={".pdf",".csv",".xlsx",".xls",".txt",".md",".json",".docx"}
MAX_BYTES=int(os.getenv("DEALLENS_MAX_UPLOAD_BYTES",25*1024*1024))

@router.post("/documents",status_code=201)
async def upload_document(file: UploadFile=File(...)):
    original=Path(file.filename or "upload").name
    ext=Path(original).suffix.lower()
    if ext not in ALLOWED: raise HTTPException(415,f"Unsupported file type {ext or '(none)'}")
    data=await file.read()
    if not data: raise HTTPException(400,"Uploaded file is empty")
    if len(data)>MAX_BYTES: raise HTTPException(413,f"File exceeds {MAX_BYTES} byte limit")
    doc_id=str(uuid.uuid4())
    UPLOAD_DIR.mkdir(parents=True,exist_ok=True)
    path=UPLOAD_DIR/f"{doc_id}{ext}"
    path.write_bytes(data)
    # Register only metadata; raw documents stay in the upload directory.
    jobs.register_document(doc_id,original,str(path))
    return {"document_id":doc_id,"filename":original,"bytes":len(data),"status":"uploaded",
            "privacy_note":"Text extracted for analysis is redacted before job results are persisted."}

class AnalysisRequest(BaseModel):
    document_ids:list[str]=Field(default_factory=list)
    data_dir:str="data/raw"

@router.post("/analyses",status_code=202)
def start_analysis(payload:AnalysisRequest,background_tasks:BackgroundTasks):
    paths=[]
    for doc_id in payload.document_ids:
        doc=jobs.get_document(doc_id)
        if not doc: raise HTTPException(404,f"Unknown document_id: {doc_id}")
        paths.append(doc["path"])
    job_id=str(uuid.uuid4()); jobs.create(job_id)
    # Use Celery when configured and reachable; retain a functional local fallback.
    if os.getenv("DEALLENS_USE_CELERY","").lower() in {"1","true","yes"}:
        try:
            from backend.workers.tasks import analyze_task
            if analyze_task is None: raise RuntimeError("Celery is not installed")
            analyze_task.delay(job_id,payload.data_dir,paths)
        except Exception:
            background_tasks.add_task(_run,job_id,payload.data_dir,paths)
    else:
        background_tasks.add_task(_run,job_id,payload.data_dir,paths)
    return {"analysis_id":job_id,"status":"queued","status_url":f"/api/v1/analyses/{job_id}",
            "findings_url":f"/api/v1/analyses/{job_id}/findings"}

def _run(job_id,data_dir,paths):
    from backend.workers.tasks import process_analysis
    process_analysis(job_id,data_dir,paths)

@router.get("/analyses/{analysis_id}")
def analysis_status(analysis_id:str):
    job=jobs.get(analysis_id)
    if not job: raise HTTPException(404,"Analysis not found")
    return {k:v for k,v in job.items() if k!="result"}

@router.get("/analyses/{analysis_id}/findings")
def analysis_findings(analysis_id:str):
    job=jobs.get(analysis_id)
    if not job: raise HTTPException(404,"Analysis not found")
    if job["status"] in {"queued","running"}: return {"analysis_id":analysis_id,"status":job["status"],"findings":[]}
    if job["status"]=="failed": raise HTTPException(500,job["error"] or "Analysis failed")
    result=job["result"] or {}
    return {"analysis_id":analysis_id,"status":job["status"],"findings":result.get("findings",[]),
            "summary":{"documents_processed":result.get("documents_processed",0),"n_findings":len(result.get("findings",[]))}}

class RedactRequest(BaseModel):
    text:str

@router.post("/privacy/redact")
def redact(payload:RedactRequest):
    return redact_text(payload.text)

class LocalModelRequest(BaseModel):
    prompt:str
    system_instruction:str=""
    model:str|None=None

@router.post("/models/ollama/generate")
def ollama_generate(payload:LocalModelRequest):
    try: return generate_local(payload.prompt,payload.system_instruction,payload.model)
    except RuntimeError as exc: raise HTTPException(503,str(exc)) from exc
