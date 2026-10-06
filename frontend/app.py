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

st.set_page_config(page_title="DealLens AI", page_icon="🔎", layout="wide")
st.title("DealLens AI")
st.caption("Evidence-first M&A due-diligence intelligence. Findings are investigation flags, not conclusions.")

with st.sidebar:
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
m[0].metric("Findings", len(findings))
m[1].metric("High", severity_counts.get("High",0))
m[2].metric("Medium", severity_counts.get("Medium",0))
m[3].metric("Evidence files", len(pipeline.DOCS["files"]))

tabs=st.tabs(["Overview","Risk Intelligence","Financials","Dependencies","Scenario Lab","Evidence Search","AI Investigation","AI Tools","Evaluation"])

with tabs[0]:
    st.subheader("Investigation overview")
    for f in findings[:8]:
        with st.expander(f"{f['severity']} · #{f['rank']} · {f['title']}"):
            st.write(f['detail'])
            st.caption(f"Evidence: {f['evidence']}")
            st.write("Management questions")
            for q in f['questions']: st.write("- "+q)
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

with tabs[8]:
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
