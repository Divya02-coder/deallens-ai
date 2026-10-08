"""DealLens AI - native Streamlit investigation dashboard.
Run from project root: python -m streamlit run frontend/app.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.services import pipeline
from risk_engine.findings import to_markdown
from finance_engine.scenarios import Base, Scenario, run as run_scenario, sensitivity

st.set_page_config(page_title="DealLens AI | M&A Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --muted:#8ea0b8; --line:#263447; --accent:#62d9c5; }
.stApp { background:radial-gradient(circle at 80% -10%,rgba(98,217,197,.10),transparent 32%),linear-gradient(180deg,#080d14,#0b1119 55%,#080d14); }
.block-container { max-width:1450px; padding-top:1.5rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#0a1018; border-right:1px solid var(--line); }
.hero { padding:28px 30px; border:1px solid #243447; border-radius:22px; background:linear-gradient(135deg,rgba(19,31,45,.96),rgba(12,19,29,.92)); box-shadow:0 18px 55px rgba(0,0,0,.22); margin-bottom:18px; }
.eyebrow { color:var(--accent); text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
.hero h1 { font-size:2.45rem; line-height:1.05; margin:.35rem 0 .65rem; letter-spacing:-.04em; }
.hero p { color:#9eacc0; max-width:900px; margin:0; font-size:1rem; }
.pill { display:inline-block; padding:5px 10px; border-radius:999px; background:#122b2a; border:1px solid #23524d; color:#8be8d9; font-size:.72rem; margin:12px 6px 0 0; }
.section-head { display:flex; align-items:center; justify-content:space-between; margin:22px 0 12px; }
.section-head h3 { margin:0; }
.section-head span { color:var(--muted); font-size:.8rem; }
.metric-card { background:linear-gradient(145deg,#111b28,#0e1621); border:1px solid var(--line); border-radius:16px; padding:17px 18px; min-height:98px; }
.metric-label { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { font-size:1.7rem; font-weight:750; margin-top:5px; }
.metric-delta { color:#79ddcb; font-size:.72rem; margin-top:3px; }
.risk-card { border:1px solid var(--line); border-radius:15px; padding:16px; background:#101824; margin-bottom:12px; }
.risk-high { border-left:4px solid #ef7777; } .risk-medium { border-left:4px solid #e8bd62; } .risk-low { border-left:4px solid #62d9c5; }
[data-testid="stTabs"] button { color:#91a2b8; font-weight:600; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#8de7d8; }
[data-testid="stFileUploaderDropzone"] { border:1px dashed #395069; background:#0d151f; }
div.stButton > button { border-radius:10px; border:1px solid #31445b; background:#111c29; }
div.stButton > button[kind="primary"] { background:linear-gradient(135deg,#1b625c,#164b58); border-color:#3b8f88; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:#0f1722; }
.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#62d9c5; margin-right:6px; box-shadow:0 0 10px rgba(98,217,197,.7); }
</style>
""", unsafe_allow_html=True)
st.markdown("""<div class="hero"><div class="eyebrow">AI-Powered M&A Intelligence</div><h1>DealLens <span style="color:#62d9c5">AI</span></h1><p>Evidence-first due diligence that connects financial signals, contracts, dependencies and transaction anomalies into investigation-ready insights.</p><span class="pill">RAG + Agentic AI</span><span class="pill">LangGraph orchestration</span><span class="pill">Evidence traceability</span></div>""", unsafe_allow_html=True)
st.caption("◉ Investigation flags are decision-support signals, not investment conclusions.")

with st.sidebar:
    st.markdown("### ◈ DealLens")
    st.markdown('<span class="status-dot"></span><b>Workspace online</b>', unsafe_allow_html=True)
    st.caption("M&A due-diligence command center")
    st.divider()
    st.header("Deal workspace")
    uploaded = st.file_uploader("Upload evidence", type=["pdf","csv","xlsx","xls","txt","md","json","docx"], accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            key=f"uploaded_{f.name}_{f.size}"
            if not st.session_state.get(key):
                try:
                    path=pipeline.save_upload(f)
                    meta=pipeline.add_file(path)
                    st.session_state[key]=True
                    st.success(f"Indexed {f.name} ({meta['chunks_indexed']} chunks)")
                except Exception as exc:
                    st.error(f"Could not index {f.name}: {exc}")
    if st.button("Reset evidence index"):
        pipeline.reset_documents()
        for k in list(st.session_state):
            if str(k).startswith("uploaded_"): del st.session_state[k]
        st.rerun()
    st.divider()
    if st.button("Run deterministic analysis", type="primary"):
        try:
            pipeline.analyze()
            st.success("Analysis completed")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

S=pipeline.STATE
if not S:
    st.info("Generate sample data with `python data/generate_sample_data.py`, then click Run deterministic analysis. You can also upload evidence for document investigation.")
    if pipeline.DOCS["files"]:
        st.write("Indexed evidence:", pipeline.DOCS["files"])
    st.stop()

findings=S.get("findings",[])
severity_counts=pd.Series([f.get("severity") for f in findings]).value_counts().to_dict()

m=st.columns(4)
metric_data=[("Active findings",len(findings),"Deterministic risk engine"),("High severity",severity_counts.get("High",0),"Priority review"),("Medium severity",severity_counts.get("Medium",0),"Requires investigation"),("Evidence files",len(pipeline.DOCS["files"]),"Indexed for RAG")]
for col,(label,value,sub) in zip(m,metric_data):
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-delta">{sub}</div></div>',unsafe_allow_html=True)

tabs=st.tabs(["Overview","Risk Intelligence","Financials","Dependencies","Scenario Lab","Evidence Search","AI Investigation","Advanced Agent","AI Tools","Evaluation"])

with tabs[0]:
    st.markdown('<div class="section-head"><h3>Investigation overview</h3><span>Prioritized signals from the current deal workspace</span></div>', unsafe_allow_html=True)
    if findings:
        top=findings[0]
        sev=str(top.get("severity","low")).lower()
        st.markdown(f'<div class="risk-card risk-{sev}"><div class="metric-label">Top priority · {top.get("severity","Unknown")}</div><h3 style="margin:.35rem 0">{top.get("title","Risk finding")}</h3><div style="color:#a7b5c7">{top.get("detail","")}</div></div>', unsafe_allow_html=True)
    for f in findings[:8]:
        with st.expander(f"{f['severity']} · #{f['rank']} · {f['title']}"):
            st.write(f['detail'])
            st.caption(f"Evidence: {f['evidence']}")
            st.write("Management questions")
            for q in f['questions']: st.write("- "+q)
    if findings:
        st.markdown('<div class="section-head"><h3>Risk distribution</h3><span>Severity across detected findings</span></div>', unsafe_allow_html=True)
        rc=pd.DataFrame({"Severity":list(severity_counts.keys()),"Findings":list(severity_counts.values())})
        st.plotly_chart(px.bar(rc,x="Severity",y="Findings",text="Findings",title=""),use_container_width=True,config={"displayModeBar":False})
    st.download_button("Download risk report", to_markdown(findings), "deallens_report.md", "text/markdown")

with tabs[1]:
    st.subheader("Risk intelligence")
    for f in findings:
        with st.expander(f"#{f['rank']} {f['severity']} — {f['title']}"):
            st.write(f['detail']); st.json(f.get('evidence',[])); st.write("Questions:")
            for q in f.get('questions',[]): st.write("- "+q)

with tabs[2]:
    r=S['ratios']
    st.dataframe(r.set_index('year').T.style.format('{:,.2f}'), use_container_width=True)
    st.plotly_chart(px.line(r,x='year',y=['dso','dio','dpo'],markers=True,title='Working-capital days'),use_container_width=True)
    st.subheader("AI financial copilot")
    fq=st.text_input("Ask about the financials", "What financial trend should an acquirer investigate first?")
    if st.button("Run financial copilot"):
        try:
            from ai_engine.financial_copilot import FinancialCopilot
            st.markdown(FinancialCopilot().answer(fq,S))
        except Exception as exc: st.error(str(exc))

with tabs[3]:
    c1,c2=st.columns(2)
    c1.plotly_chart(px.pie(S['cust'],names='customer',values='revenue',title='Revenue concentration'),use_container_width=True)
    c2.plotly_chart(px.pie(S['sup'],names='supplier',values='procurement',title='Supplier concentration'),use_container_width=True)
    st.dataframe(S['scored_tx'].head(25),use_container_width=True)

with tabs[4]:
    f=S['fin'].sort_values('year').iloc[-1]
    base=Base(f.revenue,f.cogs,f.operating_expenses,f.depreciation,f.interest_expense,f.capex,f.total_debt,f.interest_expense+S['debt']['principal'].sum()*0.2)
    a,b,c=st.columns(3); rc=a.slider('Revenue change %',-40,40,-15)/100; oc=b.slider('Opex change %',-20,30,8)/100; lost=c.slider('Lost customer revenue (Cr)',0,100,0)*1e7
    sc=Scenario(revenue_change=rc,opex_change=oc,lost_customer_revenue=lost); o=run_scenario(base,sc)
    mm=st.columns(4); mm[0].metric('Revenue (Cr)',f"{o['revenue']/1e7:,.0f}"); mm[1].metric('EBITDA (Cr)',f"{o['ebitda']/1e7:,.0f}"); mm[2].metric('FCF (Cr)',f"{o['free_cash_flow']/1e7:,.0f}"); mm[3].metric('Debt/EBITDA',f"{o['debt_to_ebitda']:.2f}x")
    sens=pd.DataFrame(sensitivity(base,sc,'revenue_change',[x/100 for x in range(-30,31,5)]))
    st.plotly_chart(px.line(sens,x='revenue_change',y='ebitda',title='EBITDA sensitivity'),use_container_width=True)

with tabs[5]:
    st.subheader("Evidence search")
    if not pipeline.DOCS['index']:
        st.info("Upload a PDF, CSV, Excel, TXT, Markdown, JSON or DOCX file first.")
    else:
        q=st.text_input('Search indexed evidence')
        if q:
            for h in pipeline.DOCS['index'].search(q,8):
                st.markdown(f"**{h['source']} · page {h.get('page',1)} · {h.get('location','')} · score {h['score']:.2f}**")
                st.caption(h['text'][:1000])

with tabs[6]:
    st.subheader("Agentic investigation")
    q=st.text_area('Investigation question','Find the top financial risks an acquirer should investigate and explain the evidence.')
    if st.button('Investigate with tools'):
        try:
            from agent.agent import investigate
            from agent.tools import Toolbox
            result=investigate(q,Toolbox(S,pipeline.DOCS['index']))
            st.markdown(result)
        except Exception as exc: st.error(str(exc))

with tabs[7]:
    st.subheader("Advanced agentic due diligence")
    st.caption("LangGraph orchestrates planning → specialist investigations → evidence verification → synthesis. The deterministic risk engine remains unchanged.")
    aq=st.text_area("Due-diligence question", "Investigate the most important risks an acquirer should focus on, connect related customer/contract/debt signals, and cite the evidence.", key="advanced_question")
    if st.button("Run advanced investigation", type="primary"):
        try:
            from advanced_agent.workflow import run_due_diligence
            from agent.tools import Toolbox
            result=run_due_diligence(aq, Toolbox(S,pipeline.DOCS['index']))
            st.success("Investigation workflow completed")
            c1,c2=st.columns(2)
            c1.metric("Planned domains", len(result.get("plan", [])))
            c2.metric("Verified findings", len(result.get("verified", [])))
            st.write("**Plan:**", ", ".join(result.get("plan", [])))
            st.markdown(result.get("report", "No report returned."))
            with st.expander("Specialist observations"):
                st.json(result.get("observations", {}))
            with st.expander("Verification trail"):
                st.json(result.get("verified", []))
        except Exception as exc:
            st.error(str(exc))

with tabs[8]:
    st.subheader("AI evidence tools")
    tool_tab1,tool_tab2,tool_tab3=st.tabs(['Schema Agent','Anomaly Explainer','Evidence Verifier'])
    with tool_tab1:
        st.write('Upload a CSV/Excel file and inspect its deterministic profile plus AI interpretation.')
        sf=st.file_uploader('CSV/Excel for schema analysis',type=['csv','xlsx','xls'],key='schema_file')
        if sf and st.button('Analyze schema'):
            try:
                import io
                if sf.name.lower().endswith('.csv'): df=pd.read_csv(io.BytesIO(sf.getvalue()))
                else: df=pd.read_excel(io.BytesIO(sf.getvalue()),sheet_name=0)
                from ai_engine.schema_agent import SchemaAgent
                st.json(SchemaAgent().infer(df))
            except Exception as exc: st.error(str(exc))
    with tool_tab2:
        if 'scored_tx' in S:
            st.dataframe(S['scored_tx'].sort_values('anomaly_score',ascending=False).head(15),use_container_width=True)
            if st.button('Explain unusual transactions'):
                try:
                    from ai_engine.anomaly_explainer import AnomalyExplainer
                    st.markdown(AnomalyExplainer().explain(S['scored_tx'].sort_values('anomaly_score',ascending=False).head(15)))
                except Exception as exc: st.error(str(exc))
    with tool_tab3:
        claim=st.text_area('Claim to verify','The target has a material customer concentration risk.')
        if st.button('Verify claim'):
            try:
                evidence=pipeline.DOCS['index'].search(claim,5) if pipeline.DOCS['index'] else []
                from ai_engine.evidence_verifier import EvidenceVerifier
                st.json(EvidenceVerifier().verify(claim,evidence))
                st.write('Retrieved evidence')
                for e in evidence: st.caption(f"{e['source']} p.{e.get('page',1)} — {e['text'][:500]}")
            except Exception as exc: st.error(str(exc))

with tabs[9]:
    st.subheader("AI evaluation checks")
    st.write("Small regression checks for evidence-first behavior.")
    if st.button('Run evaluation'):
        try:
            from ai_engine.evaluator import evaluate
            from agent.agent import investigate
            from agent.tools import Toolbox
            rows=evaluate(lambda q: investigate(q,Toolbox(S,pipeline.DOCS['index']),max_steps=4))
            st.dataframe(pd.DataFrame(rows),use_container_width=True)
        except Exception as exc: st.error(str(exc))
