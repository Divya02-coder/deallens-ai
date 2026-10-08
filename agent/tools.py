"""Deterministic investigation tools exposed to the Gemini agent."""
from __future__ import annotations
import json
from finance_engine.ratios import compute_ratios
from dependency_engine.concentration import concentration
from graph_engine.knowledge_graph import build_graph, query


class Toolbox:
    def __init__(self, state: dict, doc_index=None, company: str = "Target Co"):
        self.s, self.docs, self.company = state, doc_index, company
        self.graph = build_graph(company, state.get("cust"), state.get("sup"), state.get("debt"), state.get("contracts"))

    def list_findings(self, category: str | None = None):
        return [x for x in self.s.get("findings", []) if not category or x.get("category", "").lower() == category.lower()]

    def calculate_ratio(self, name: str, year: int | None = None):
        r = self.s.get("ratios")
        if r is None or name not in r.columns:
            return {"error": f"unknown ratio. available: {list(r.columns) if r is not None else []}"}
        row = r if year is None else r[r["year"] == year]
        return row[["year", name]].to_dict("records")

    def analyze_customers(self):
        c = concentration(self.s["cust"], "customer", "revenue")
        return {k: c[k] for k in ("top1_name", "top1", "top_n", "hhi")}

    def analyze_suppliers(self):
        c = concentration(self.s["sup"], "supplier", "procurement")
        return {k: c[k] for k in ("top1_name", "top1", "top_n", "hhi")}

    def query_knowledge_graph(self, rel: str | None = None, min_share: float = 0.0):
        return query(self.graph, self.company, rel, min_share)

    def analyze_contracts(self):
        """Return structured contract exposure signals without changing risk-engine logic."""
        contracts = self.s.get("contracts")
        if contracts is None or contracts.empty:
            return {"contracts": [], "count": 0}
        rows = []
        today = __import__("pandas").Timestamp.today().normalize()
        for _, r in contracts.iterrows():
            end = __import__("pandas").to_datetime(r.get("end_date"), errors="coerce")
            days = int((end - today).days) if __import__("pandas").notna(end) else None
            rows.append({
                "contract_id": r.get("contract_id"),
                "counterparty": r.get("counterparty"),
                "type": r.get("type"),
                "end_date": str(r.get("end_date")),
                "days_to_expiry": days,
                "change_of_control": bool(r.get("change_of_control", False)),
                "value": float(r.get("value", 0)) if str(r.get("value", "")) not in ("", "nan") else None,
            })
        return {"contracts": rows, "count": len(rows)}

    def search_documents(self, query_text: str, k: int = 5):
        if not self.docs:
            return {"error": "No documents indexed."}
        return {"query": query_text, "results": self.docs.search(query_text, k)}

    def verify_claim(self, claim: str, k: int = 5):
        results = self.docs.search(claim, k) if self.docs else []
        from ai_engine.evidence_verifier import EvidenceVerifier
        return EvidenceVerifier().verify(claim, results)

    def get_transaction_anomalies(self, limit: int = 10):
        tx = self.s.get("scored_tx")
        if tx is None:
            return {"error": "Transaction analysis is not available."}
        cols = [c for c in ["date", "vendor", "amount", "anomaly_score"] if c in tx.columns]
        return tx.sort_values("anomaly_score", ascending=False).head(limit)[cols].to_dict("records")

    def run(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None)
        if name.startswith("_") or fn is None:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return json.dumps(fn(**(args or {})), default=str)[:14000]
        except Exception as e:
            return json.dumps({"error": str(e)})


TOOL_SCHEMAS = [
    {"name":"list_findings","description":"List deterministic risk findings, optionally by category.","parameters":{"type":"OBJECT","properties":{"category":{"type":"STRING"}}}},
    {"name":"calculate_ratio","description":"Retrieve a computed financial ratio by year.","parameters":{"type":"OBJECT","properties":{"name":{"type":"STRING"},"year":{"type":"INTEGER"}},"required":["name"]}},
    {"name":"analyze_customers","description":"Return customer concentration metrics.","parameters":{"type":"OBJECT","properties":{}}},
    {"name":"analyze_suppliers","description":"Return supplier concentration metrics.","parameters":{"type":"OBJECT","properties":{}}},
    {"name":"query_knowledge_graph","description":"Query customer, supplier and debt relationships.","parameters":{"type":"OBJECT","properties":{"rel":{"type":"STRING"},"min_share":{"type":"NUMBER"}}}},
    {"name":"analyze_contracts","description":"Inspect contract expiry, value and change-of-control exposure.","parameters":{"type":"OBJECT","properties":{}}},
    {"name":"search_documents","description":"Search uploaded evidence and return source/page/location metadata.","parameters":{"type":"OBJECT","properties":{"query_text":{"type":"STRING"},"k":{"type":"INTEGER"}},"required":["query_text"]}},
    {"name":"verify_claim","description":"Check a claim against retrieved document evidence.","parameters":{"type":"OBJECT","properties":{"claim":{"type":"STRING"},"k":{"type":"INTEGER"}},"required":["claim"]}},
    {"name":"get_transaction_anomalies","description":"Return highest-scoring unusual transactions from the deterministic anomaly model.","parameters":{"type":"OBJECT","properties":{"limit":{"type":"INTEGER"}}}},
]
