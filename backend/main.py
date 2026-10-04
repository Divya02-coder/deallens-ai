"""STEP 13 - FastAPI backend.  Run: uvicorn backend.main:app --reload"""
import os, shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from backend.services import pipeline
from risk_engine.findings import to_markdown
from finance_engine.scenarios import Base, Scenario, run as run_scenario

app = FastAPI(title="DealLens AI")
UPLOADS = Path("data/raw"); UPLOADS.mkdir(parents=True, exist_ok=True)
ALLOWED = {".csv", ".xlsx", ".pdf"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"unsupported type {ext}")
    dest = UPLOADS / Path(file.filename).name          # .name blocks path traversal
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    info = {"saved": dest.name}
    if ext == ".pdf":
        info["chunks_indexed"] = pipeline.add_pdf(str(dest))
    return info


@app.post("/analyze")
def analyze():
    res = pipeline.analyze()
    return {"n_findings": len(res["findings"]), "findings": res["findings"]}


@app.get("/report")
def report():
    if not pipeline.STATE:
        raise HTTPException(409, "run /analyze first")
    return {"markdown": to_markdown(pipeline.STATE["findings"])}


class ScenarioIn(BaseModel):
    revenue_change: float = 0.0
    opex_change: float = 0.0
    lost_customer_revenue: float = 0.0
    synergy: float = 0.0


@app.post("/scenario")
def scenario(s: ScenarioIn):
    if not pipeline.STATE:
        raise HTTPException(409, "run /analyze first")
    f = pipeline.STATE["fin"].sort_values("year").iloc[-1]
    base = Base(revenue=f.revenue, cogs=f.cogs, opex=f.operating_expenses, depreciation=f.depreciation,
                interest=f.interest_expense, capex=f.capex, debt=f.total_debt,
                debt_service=f.interest_expense + pipeline.STATE["debt"]["principal"].sum() * 0.2)
    return run_scenario(base, Scenario(**s.model_dump()))


class Q(BaseModel):
    question: str


@app.post("/ask")
def ask(q: Q):
    if not pipeline.STATE:
        raise HTTPException(409, "run /analyze first")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "Set ANTHROPIC_API_KEY to enable the investigation agent")
    from agent.agent import investigate
    from agent.tools import Toolbox
    return {"answer": investigate(q.question, Toolbox(pipeline.STATE, pipeline.DOCS["index"]))}
