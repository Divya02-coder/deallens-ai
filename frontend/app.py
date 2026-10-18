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
:root { --ink:#172b4d; --muted:#64748b; --line:#e2e8f0; --accent:#147d75; --surface:#ffffff; }
.stApp { background:#f5f7fb; color:#172b4d; }
.block-container { max-width:1480px; padding-top:1.4rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#101d32; border-right:1px solid #263650; }
[data-testid="stSidebar"] * { color:#e8eef7; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background:#172740; border:1px dashed #526783; }
[data-testid="stSidebar"] button { border-radius:9px; }
.hero { padding:32px 36px; border:1px solid #dce5ee; border-radius:20px; background:linear-gradient(120deg,#ffffff 0%,#edf7f6 100%); box-shadow:0 8px 28px rgba(15,35,60,.06); margin-bottom:18px; }
.eyebrow { color:#147d75; text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:800; }
.hero h1 { color:#132640; font-size:2.5rem; line-height:1.08; margin:.45rem 0 .7rem; letter-spacing:-.045em; }
.hero p { color:#52647b; max-width:850px; margin:0; font-size:1.02rem; line-height:1.65; }
.pill { display:inline-block; padding:5px 10px; border-radius:7px; background:#e4f4f1; border:1px solid #c4e7e1; color:#176c65; font-size:.72rem; margin:14px 6px 0 0; font-weight:650; }
.section-head { display:flex; align-items:center; justify-content:space-between; margin:24px 0 12px; }
.section-head h3 { margin:0; color:#172b4d; }
.section-head span { color:var(--muted); font-size:.8rem; }
.metric-card { background:#fff; border:1px solid #e1e8f0; border-radius:14px; padding:18px 19px; min-height:106px; box-shadow:0 3px 12px rgba(20,40,70,.035); }
.metric-label { color:#64748b; font-size:.72rem; text-transform:uppercase; letter-spacing:.075em; font-weight:700; }
.metric-value { color:#142843; font-size:1.85rem; font-weight:760; margin-top:7px; }
.metric-delta { color:#147d75; font-size:.75rem; margin-top:4px; }
.risk-card { border:1px solid #e0e7ef; border-radius:14px; padding:18px; background:#fff; margin-bottom:12px; box-shadow:0 3px 12px rgba(20,40,70,.035); }
.risk-high { border-left:4px solid #d94b55; } .risk-medium { border-left:4px solid #d99a21; } .risk-low { border-left:4px solid #159789; }
[data-testid="stTabs"] { background:transparent; }
[data-testid="stTabs"] button { color:#64748b; font-weight:650; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#147d75; border-bottom-color:#147d75; }
[data-testid="stFileUploaderDropzone"] { border:1px dashed #b8c8d9; background:#f8fafc; border-radius:10px; }
div.stButton > button, div.stDownloadButton > button { border-radius:9px; border:1px solid #d5dfeb; background:#fff; color:#243b57; font-weight:600; min-height:2.55rem; }
div.stButton > button:hover, div.stDownloadButton > button:hover { border-color:#147d75; color:#126b65; }
div.stButton > button[kind="primary"] { background:#147d75; border-color:#147d75; color:light blue; }
div.stButton > button[kind="primary"]:hover { background:#106b64; color:light blue; }
[data-testid="stExpander"] { border:1px solid #e0e7ef; border-radius:11px; background:#fff; }
[data-testid="stDataFrame"] { border:1px solid #e0e7ef; border-radius:10px; overflow:hidden; }
.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#45c2a9; margin-right:7px; box-shadow:0 0 0 3px rgba(69,194,169,.14); }
.stAlert { border-radius:10px; }
small, .stCaption { color:#64748b; }
</style>
""", unsafe_allow_html=True)
st.markdown("""<div class="hero"><div class="eyebrow">M&A · Risk & Due Diligence</div><h1>DealLens <span style="color:#147d75">AI</span></h1><p>A clearer view of deal risk. Bring financial data and deal documents together to surface unusual activity, concentration risks, and evidence-backed questions for your diligence team.</p><span class="pill">Evidence-backed insights</span><span class="pill">Financial & transaction risk</span><span class="pill">Investigation workspace</span></div>""", unsafe_allow_html=True)
st.caption("Decision-support only: findings should be reviewed alongside source evidence and professional judgement.")

with st.sidebar:
    st.markdown("###  DealLens")
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
    if st.button("Reset current analysis", help="Clear analysis results while keeping uploaded and indexed evidence"):
        pipeline.STATE.clear()
        for k in ("advanced_question",):
            st.session_state.pop(k, None)
        st.rerun()
    if st.button("Reset evidence index", help="Remove indexed documents from the current workspace"):
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
