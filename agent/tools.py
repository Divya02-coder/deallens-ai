"""STEP 12a - Tools the agent may call. The LLM never does the math; it calls these."""
import json
import pandas as pd
from finance_engine.ratios import compute_ratios
from dependency_engine.concentration import concentration
from graph_engine.knowledge_graph import build_graph, query


class Toolbox:
    def __init__(self, state: dict, doc_index=None, company: str = "Target Co"):
        self.s, self.docs, self.company = state, doc_index, company
        self.graph = build_graph(company, state["cust"], state["sup"], state["debt"], state["contracts"])

    # ---- tool implementations ----
    def list_findings(self, category: str | None = None):
        f = self.s["findings"]
        return [x for x in f if not category or x["category"].lower() == category.lower()]

    def calculate_ratio(self, name: str, year: int | None = None):
        r = self.s["ratios"]
        if name not in r.columns:
            return {"error": f"unknown ratio. available: {list(r.columns)}"}
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

    def search_documents(self, query_text: str, k: int = 5):
        return self.docs.search(query_text, k) if self.docs else {"error": "no documents indexed"}

    def run(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None)
        if name.startswith("_") or fn is None:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return json.dumps(fn(**args), default=str)[:12000]
        except Exception as e:  # return errors to the model instead of crashing
            return json.dumps({"error": str(e)})


TOOL_SCHEMAS = [
    {"name": "list_findings", "description": "List risk findings from the deterministic engines, optionally by category.",
     "input_schema": {"type": "object", "properties": {"category": {"type": "string"}}}},
    {"name": "calculate_ratio", "description": "Get a computed financial ratio (e.g. dso, ebitda_margin, debt_to_ebitda) per year.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "year": {"type": "integer"}}, "required": ["name"]}},
    {"name": "analyze_customers", "description": "Customer concentration metrics.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "analyze_suppliers", "description": "Supplier concentration metrics.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "query_knowledge_graph", "description": "Query company relationships. rel: earns_revenue_from | purchases_from | owes.",
     "input_schema": {"type": "object", "properties": {"rel": {"type": "string"}, "min_share": {"type": "number"}}}},
    {"name": "search_documents", "description": "Search uploaded PDFs; returns text with source and page for citations.",
     "input_schema": {"type": "object", "properties": {"query_text": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query_text"]}},
]
