"""LangGraph-based DealLens investigation workflow.

This layer orchestrates the existing deterministic engines; it does not replace them.
That separation keeps benchmark behavior stable while adding agentic planning,
specialist investigations, evidence verification and report synthesis.
"""
from __future__ import annotations

import json
from typing import Any, TypedDict

from agent.tools import Toolbox


DOMAINS = ("financial", "dependencies", "contracts", "transactions", "evidence")


class InvestigationState(TypedDict, total=False):
    question: str
    plan: list[str]
    observations: dict[str, Any]
    evidence: list[dict]
    verified: list[dict]
    report: str


def _fallback_plan(question: str) -> list[str]:
    q = question.lower()
    plan: list[str] = []
    if any(x in q for x in ("cash", "revenue", "ebitda", "margin", "financial", "profit", "debt")):
        plan.append("financial")
    if any(x in q for x in ("customer", "supplier", "dependency", "concentration")):
        plan.append("dependencies")
    if any(x in q for x in ("contract", "renew", "expiry", "change of control", "clause")):
        plan.append("contracts")
    if any(x in q for x in ("transaction", "payment", "anomal", "duplicate")):
        plan.append("transactions")
    if any(x in q for x in ("document", "evidence", "verify", "prove", "source")):
        plan.append("evidence")
    return plan or list(DOMAINS)


def _plan(question: str) -> list[str]:
    """Use the LLM as a planner, with a deterministic safe fallback."""
    prompt = f"""Choose the investigation domains needed to answer this M&A due-diligence question.
Return JSON only: {{\"domains\":[\"financial\",\"dependencies\",\"contracts\",\"transactions\",\"evidence\"]}}
Only use these exact domains: {', '.join(DOMAINS)}.
Question: {question}"""
    try:
        from ai_engine.llm import GeminiLLM
        raw = GeminiLLM().generate(prompt, "You are a conservative investigation planner. JSON only.", 500)
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        selected = [x for x in data.get("domains", []) if x in DOMAINS]
        return selected or _fallback_plan(question)
    except Exception:
        return _fallback_plan(question)


def _financial(tb: Toolbox) -> dict:
    ratios = tb.s.get("ratios")
    if ratios is None or ratios.empty:
        return {"error": "Financial ratios unavailable."}
    latest = ratios.sort_values("year").iloc[-1]
    keys = ["revenue_growth", "ebitda_margin", "cash_conversion", "free_cash_flow", "debt_to_ebitda", "interest_coverage"]
    return {k: (None if str(latest.get(k)) == "nan" else float(latest.get(k))) for k in keys if k in latest}


def _dependencies(tb: Toolbox) -> dict:
    return {"customers": tb.analyze_customers(), "suppliers": tb.analyze_suppliers()}


def _contracts(tb: Toolbox) -> dict:
    return tb.analyze_contracts()


def _transactions(tb: Toolbox) -> dict:
    return {"top_anomalies": tb.get_transaction_anomalies(8)}


def _evidence(tb: Toolbox, question: str) -> dict:
    return {"results": tb.search_documents(question, 6) if tb.docs else [], "indexed": bool(tb.docs)}


def _specialists(state: InvestigationState, toolbox: Toolbox) -> dict:
    observations: dict[str, Any] = {}
    handlers = {
        "financial": _financial,
        "dependencies": _dependencies,
        "contracts": _contracts,
        "transactions": _transactions,
    }
    for domain in state.get("plan", []):
        if domain in handlers:
            try:
                observations[domain] = handlers[domain](toolbox)
            except Exception as exc:
                observations[domain] = {"error": str(exc)}
        elif domain == "evidence":
            observations[domain] = _evidence(toolbox, state["question"])
    return observations


def _verification(state: InvestigationState, toolbox: Toolbox) -> list[dict]:
    verified = []
    for finding in toolbox.list_findings()[:8]:
        evidence = finding.get("evidence", [])
        if not evidence:
            continue
        # Deterministic finding evidence is already source-addressed. If document
        # evidence exists, additionally ask the existing verifier to test the claim.
        item = {"finding": finding.get("title"), "status": "evidence_present", "evidence": evidence}
        if toolbox.docs:
            try:
                claim = finding.get("detail", finding.get("title", ""))
                item["document_check"] = toolbox.verify_claim(claim, 4)
            except Exception as exc:
                item["document_check"] = {"error": str(exc)}
        verified.append(item)
    return verified


def _synthesize(state: InvestigationState) -> str:
    prompt = f"""Prepare an evidence-first M&A due-diligence response.
Question: {state['question']}
Investigation plan: {state.get('plan', [])}
Specialist observations: {json.dumps(state.get('observations', {}), default=str)[:18000]}
Verified findings: {json.dumps(state.get('verified', []), default=str)[:12000]}

Rules:
- Separate facts, signals and recommendations.
- Never invent evidence.
- Prioritize the highest-impact risks.
- Mention source names when available.
- End with 3 concrete management questions.
- If evidence is insufficient, explicitly say so.
"""
    try:
        from ai_engine.llm import GeminiLLM
        return GeminiLLM().generate(prompt, "You are a senior M&A due-diligence analyst. Be precise and evidence-first.", 3000)
    except Exception as exc:
        # The workflow remains useful without an API key.
        return "LLM synthesis unavailable: " + str(exc) + "\n\n" + json.dumps(state.get("verified", []), default=str, indent=2)


def build_workflow(toolbox: Toolbox):
    """Build the graph lazily so the base project works without LangGraph installed."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph is not installed. Run: pip install -U langgraph") from exc

    graph = StateGraph(InvestigationState)

    def planner(state: InvestigationState):
        return {"plan": _plan(state["question"])}

    def specialists(state: InvestigationState):
        return {"observations": _specialists(state, toolbox)}

    def verifier(state: InvestigationState):
        return {"verified": _verification(state, toolbox)}

    def synthesizer(state: InvestigationState):
        return {"report": _synthesize(state)}

    graph.add_node("planner", planner)
    graph.add_node("specialists", specialists)
    graph.add_node("verifier", verifier)
    graph.add_node("synthesizer", synthesizer)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "specialists")
    graph.add_edge("specialists", "verifier")
    graph.add_edge("verifier", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()


def run_due_diligence(question: str, toolbox: Toolbox) -> InvestigationState:
    """Run the advanced investigation graph and return its complete state."""
    app = build_workflow(toolbox)
    return app.invoke({"question": question})
