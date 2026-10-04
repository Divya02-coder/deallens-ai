"""STEP 8 - Knowledge graph with NetworkX (migrate to Neo4j later)."""
import networkx as nx
import pandas as pd


def build_graph(company: str, cust: pd.DataFrame, sup: pd.DataFrame, debt: pd.DataFrame, contracts: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node(company, kind="company")
    rt, pt = cust["revenue"].sum(), sup["procurement"].sum()
    for _, r in cust.iterrows():
        g.add_node(r["customer"], kind="customer")
        g.add_edge(company, r["customer"], rel="earns_revenue_from", share=float(r["revenue"] / rt), amount=float(r["revenue"]))
    for _, r in sup.iterrows():
        g.add_node(r["supplier"], kind="supplier")
        g.add_edge(company, r["supplier"], rel="purchases_from", share=float(r["procurement"] / pt), amount=float(r["procurement"]))
    for _, r in debt.iterrows():
        g.add_node(r["lender"], kind="lender")
        g.add_edge(company, r["lender"], rel="owes", amount=float(r["principal"]), maturity=str(r["maturity_date"]))
    for _, r in contracts.iterrows():
        cid = f"Contract {r['contract_id']}"
        g.add_node(cid, kind="contract", end_date=str(r["end_date"]), change_of_control=bool(r["change_of_control"]))
        if r["counterparty"] in g:
            g.add_edge(r["counterparty"], cid, rel="governed_by")
    return g


def query(g: nx.DiGraph, company: str, rel: str | None = None, min_share: float = 0.0) -> list[dict]:
    rows = []
    for _, n, d in g.out_edges(company, data=True):
        if (rel is None or d["rel"] == rel) and d.get("share", 1.0) >= min_share:
            rows.append({"entity": n, "kind": g.nodes[n]["kind"], **d})
    return sorted(rows, key=lambda x: -x.get("amount", 0))


def contracts_of(g: nx.DiGraph, entity: str) -> list[dict]:
    return [{"contract": n, **g.nodes[n]} for _, n, d in g.out_edges(entity, data=True) if d["rel"] == "governed_by"]
