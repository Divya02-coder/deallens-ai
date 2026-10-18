"""Persistent SQLite job and upload registry shared by API and worker processes."""
from __future__ import annotations
import json, os, sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("DEALLENS_JOBS_DB", "data/jobs.sqlite3"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _connect():
    conn=sqlite3.connect(DB_PATH,timeout=30)
    conn.row_factory=sqlite3.Row
    return conn

def now(): return datetime.now(timezone.utc).isoformat()

def init_db():
    with _connect() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY,status TEXT NOT NULL,created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,error TEXT,result_json TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY,filename TEXT NOT NULL,path TEXT NOT NULL,
          uploaded_at TEXT NOT NULL,redacted INTEGER NOT NULL DEFAULT 1)""")

def create(job_id):
    init_db(); t=now()
    with _connect() as c:
        c.execute("INSERT INTO jobs(id,status,created_at,updated_at) VALUES(?,?,?,?)",(job_id,"queued",t,t))

def update(job_id,status,result=None,error=None):
    init_db()
    with _connect() as c:
        c.execute("UPDATE jobs SET status=?,updated_at=?,error=?,result_json=? WHERE id=?",
                  (status,now(),error,json.dumps(result,default=str) if result is not None else None,job_id))

def get(job_id):
    init_db()
    with _connect() as c: row=c.execute("SELECT * FROM jobs WHERE id=?",(job_id,)).fetchone()
    if not row: return None
    d=dict(row); raw=d.pop("result_json"); d["result"]=json.loads(raw) if raw else None
    return d

def register_document(doc_id,filename,path):
    init_db()
    with _connect() as c:
        c.execute("INSERT INTO documents(id,filename,path,uploaded_at) VALUES(?,?,?,?)",
                  (doc_id,filename,path,now()))

def get_document(doc_id):
    init_db()
    with _connect() as c: row=c.execute("SELECT * FROM documents WHERE id=?",(doc_id,)).fetchone()
    return dict(row) if row else None

def list_documents():
    init_db()
    with _connect() as c: rows=c.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    return [dict(r) for r in rows]
