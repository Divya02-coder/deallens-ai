import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dotenv import load_dotenv

from backend.services import pipeline


load_dotenv()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DealLens | M&A Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD CSS
# ============================================================

css_path = Path(__file__).parent / "style.css"

if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text()}</style>",
        unsafe_allow_html=True,
    )


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Overview"

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def clean_columns(df):
    if df is None:
        return df

    df = df.copy()

    duplicate_mask = df.columns.duplicated()

    if duplicate_mask.any():
        df = df.loc[:, ~duplicate_mask]

    return df


def latest_row(df):
    if df is None or df.empty:
        return None

    df = clean_columns(df)

    if "year" in df.columns:
        df = df.sort_values("year")

    return df.iloc[-1]


def get_state():
    return getattr(
        pipeline,
        "STATE",
        {},
    )


def get_findings():
    state = get_state()

    findings = state.get(
        "findings",
        [],
    )

    if findings is None:
        return []

    return findings


def finding_title(f):
    return (
        f.get("title")
        or f.get("name")
        or f.get("risk")
        or "Investigation finding"
    )


def finding_description(f):
    return (
        f.get("description")
        or f.get("message")
        or f.get("detail")
        or "Requires analyst investigation."
    )


def finding_severity(f):
    return str(
        f.get(
            "severity",
            "MEDIUM",
        )
    ).upper()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="deallens-header">

        <div class="brand">

            <div class="brand-mark">
                DL
            </div>

            <div>
                <div class="brand-title">
                    DealLens
                </div>

                <div class="brand-subtitle">
                    M&A Intelligence & Due-Diligence Workspace
                </div>
            </div>

        </div>

        <div class="status-pill">
            <span class="status-dot"></span>
            Intelligence Engine Online
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "### DEAL WORKSPACE"
    )

    pages = [
        "Overview",
        "Risk Intelligence",
        "Financial Intelligence",
        "Dependencies",
        "Transactions",
        "Scenario Lab",
        "Documents",
        "AI Investigation",
    ]

    for page in pages:

        if st.button(
            page,
            key=f"nav_{page}",
            width="stretch",
        ):
            st.session_state.page = page
            st.rerun()

    st.divider()

    st.markdown(
        "### DATA INGESTION"
    )

    uploaded_pdf = st.file_uploader(
        "Upload company document",
        type=["pdf"],
        help=(
            "Upload annual reports, contracts, "
            "financial statements or other PDFs."
        ),
    )

    if uploaded_pdf:

        upload_dir = ROOT / "data" / "raw"
        upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path = (
            upload_dir /
            uploaded_pdf.name
        )

        file_path.write_bytes(
            uploaded_pdf.getbuffer()
        )

        if st.button(
            "Index document",
            type="primary",
            width="stretch",
        ):

            try:

                count = pipeline.add_pdf(
                    str(file_path)
                )

                st.success(
                    f"{count} document chunks indexed."
                )

            except Exception as exc:

                st.error(
                    f"Document indexing failed: {exc}"
                )

    st.divider()

    st.caption(
        "DealLens v0.2 • Analyst Workspace"
    )


# ============================================================
# ANALYSIS BUTTON
# ============================================================

if not st.session_state.analysis_complete:

    st.markdown(
        """
        <div class="ai-panel">

            <div class="ai-label">
                FIRST-PASS DUE DILIGENCE
            </div>

            <h2>
                Understand the target company.
            </h2>

            <p>
                Run the deterministic financial, anomaly,
                dependency and risk engines before starting
                AI investigations.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Run DealLens Analysis",
        type="primary",
        width="stretch",
    ):

        with st.spinner(
            "Running financial and risk intelligence..."
        ):

            try:

                pipeline.analyze(
                    str(
                        ROOT / "data" / "raw"
                    )
                )

                st.session_state.analysis_complete = True

                st.rerun()

            except Exception as exc:

                st.error(
                    f"Analysis failed: {exc}"
                )

    st.stop()


# ============================================================
# STATE
# ============================================================

state = get_state()

findings = get_findings()


# ============================================================
# OVERVIEW
# ============================================================

if st.session_state.page == "Overview":

    st.markdown(
        '<div class="page-title">Deal Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Executive view of financial performance, '
        'risk exposure and investigation priorities.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    high_count = sum(
        finding_severity(f) == "HIGH"
        for f in findings
    )

    medium_count = sum(
        finding_severity(f) == "MEDIUM"
        for f in findings
    )

    doc_count = len(
        pipeline.DOCS.get(
            "chunks",
            [],
        )
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">
                    Total Findings
                </div>

                <div class="kpi-value">
                    {len(findings)}
                </div>

                <div class="kpi-description">
                    Issues requiring review
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">
                    High Priority
                </div>

                <div class="kpi-value">
                    {high_count}
                </div>

                <div class="kpi-description">
                    Immediate investigation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">
                    Medium Priority
                </div>

                <div class="kpi-value">
                    {medium_count}
                </div>

                <div class="kpi-description">
                    Requires analyst review
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">
                    Evidence Chunks
                </div>

                <div class="kpi-value">
                    {doc_count}
                </div>

                <div class="kpi-description">
                    Indexed document evidence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # --------------------------------------------------------
    # PRIORITY FINDINGS
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-card">

            <div class="section-title">
                Priority Investigations
            </div>

            <div class="section-description">
                Highest-value findings surfaced by
                DealLens deterministic analysis.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if not findings:

        st.info(
            "No findings are currently available."
        )

    else:

        for f in findings[:5]:

            severity = finding_severity(f)

            if severity == "HIGH":
                css_class = "risk-high"
                badge = "badge-high"

            elif severity == "LOW":
                css_class = "risk-low"
                badge = "badge-low"

            else:
                css_class = "risk-medium"
                badge = "badge-medium"

            st.markdown(
                f"""
                <div class="risk-card {css_class}">

                    <span class="risk-badge {badge}">
                        {severity}
                    </span>

                    <h3>
                        {finding_title(f)}
                    </h3>

                    <p>
                        {finding_description(f)}
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# RISK INTELLIGENCE
# ============================================================

elif st.session_state.page == "Risk Intelligence":

    st.markdown(
        '<div class="page-title">Risk Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Evidence-backed risk register for analyst review.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not findings:

        st.info(
            "No risk findings available."
        )

    else:

        for index, f in enumerate(findings, 1):

            severity = finding_severity(f)

            if severity == "HIGH":
                css_class = "risk-high"
                badge = "badge-high"

            elif severity == "LOW":
                css_class = "risk-low"
                badge = "badge-low"

            else:
                css_class = "risk-medium"
                badge = "badge-medium"

            st.markdown(
                f"""
                <div class="risk-card {css_class}">

                    <span class="risk-badge {badge}">
                        {severity}
                    </span>

                    <h3>
                        Risk #{index} — {finding_title(f)}
                    </h3>

                    <p>
                        {finding_description(f)}
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# FINANCIAL INTELLIGENCE
# ============================================================

elif st.session_state.page == "Financial Intelligence":

    st.markdown(
        '<div class="page-title">Financial Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Deterministic financial analysis and historical trends.'
        '</div>',
        unsafe_allow_html=True,
    )

    financials = state.get(
        "financials"
    )

    if isinstance(
        financials,
        pd.DataFrame,
    ):

        financials = clean_columns(
            financials
        )

        st.dataframe(
            financials,
            width="stretch",
            hide_index=True,
        )

        numeric_cols = financials.select_dtypes(
            include="number"
        ).columns.tolist()

        if (
            len(numeric_cols) >= 2
            and "year" in financials.columns
        ):

            value_col = numeric_cols[0]

            fig = px.line(
                financials,
                x="year",
                y=value_col,
                markers=True,
                title=f"{value_col.title()} Trend",
            )

            fig.update_layout(
                template="plotly_white",
                height=400,
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    else:

        st.info(
            "Financial dataset is not available "
            "in the current analysis state."
        )


# ============================================================
# DEPENDENCIES
# ============================================================

elif st.session_state.page == "Dependencies":

    st.markdown(
        '<div class="page-title">Dependency Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Customer, supplier and business relationship exposure.'
        '</div>',
        unsafe_allow_html=True,
    )

    dependency_data = state.get(
        "dependencies"
    )

    if isinstance(
        dependency_data,
        pd.DataFrame,
    ):

        st.dataframe(
            dependency_data,
            width="stretch",
            hide_index=True,
        )

    elif dependency_data:

        st.json(
            dependency_data
        )

    else:

        st.info(
            "Dependency analysis data is not available "
            "in the current state."
        )


# ============================================================
# TRANSACTIONS
# ============================================================

elif st.session_state.page == "Transactions":

    st.markdown(
        '<div class="page-title">Transaction Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Unusual transaction patterns requiring investigation.'
        '</div>',
        unsafe_allow_html=True,
    )

    transactions = state.get(
        "transactions"
    )

    if isinstance(
        transactions,
        pd.DataFrame,
    ):

        st.dataframe(
            transactions,
            width="stretch",
            hide_index=True,
        )

    elif transactions:

        st.json(
            transactions
        )

    else:

        st.info(
            "Transaction analysis is not available."
        )


# ============================================================
# SCENARIO LAB
# ============================================================

elif st.session_state.page == "Scenario Lab":

    st.markdown(
        '<div class="page-title">Scenario Lab</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Stress-test business assumptions and financial exposure.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        revenue_change = st.slider(
            "Revenue change",
            min_value=-50,
            max_value=50,
            value=0,
            step=5,
            format="%d%%",
        )

        cost_change = st.slider(
            "Operating cost change",
            min_value=-30,
            max_value=50,
            value=0,
            step=5,
            format="%d%%",
        )

    with col2:

        customer_loss = st.slider(
            "Major customer revenue loss",
            min_value=0,
            max_value=50,
            value=0,
            step=5,
            format="%d%%",
        )

        debt_change = st.slider(
            "Debt change",
            min_value=-20,
            max_value=100,
            value=0,
            step=10,
            format="%d%%",
        )

    st.markdown(
        """
        <div class="section-card">

        <div class="section-title">
            Scenario Assumptions
        </div>

        <div class="section-description">
            These controls are currently an interactive
            stress-testing layer. We will connect them to
            the full finance engine in Phase 5.
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Revenue",
        f"{100 + revenue_change}%",
        f"{revenue_change:+d}%",
    )

    s2.metric(
        "Operating Costs",
        f"{100 + cost_change}%",
        f"{cost_change:+d}%",
    )

    s3.metric(
        "Customer Exposure",
        f"{customer_loss}%",
        f"{customer_loss:+d}%",
    )

    s4.metric(
        "Debt",
        f"{100 + debt_change}%",
        f"{debt_change:+d}%",
    )


# ============================================================
# DOCUMENTS
# ============================================================

elif st.session_state.page == "Documents":

    st.markdown(
        '<div class="page-title">Document Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Semantic document search with evidence references.'
        '</div>',
        unsafe_allow_html=True,
    )

    chunks = pipeline.DOCS.get(
        "chunks",
        [],
    )

    index = pipeline.DOCS.get(
        "index"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Indexed Chunks",
        len(chunks),
    )

    c2.metric(
        "Vector Index",
        "Ready" if index else "Not initialized",
    )

    if index:

        query = st.text_input(
            "Search company evidence",
            placeholder=(
                "e.g. Which contracts expire soon?"
            ),
        )

        if query:

            results = index.search(
                query,
                k=5,
            )

            if results:

                for result in results:

                    st.markdown(
                        f"""
                        <div class="evidence-card">

                            <div class="evidence-source">
                                {result.get("source", "Unknown source")}
                            </div>

                            <div class="evidence-page">
                                Page {result.get("page", 0)}
                                · Relevance
                                {safe_float(result.get("score")):.2f}
                            </div>

                            <div class="evidence-text">
                                {result.get("text", "")}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            else:

                st.info(
                    "No relevant evidence found."
                )

    else:

        st.info(
            "Upload and index a PDF to activate "
            "semantic document search."
        )


# ============================================================
# AI INVESTIGATION
# ============================================================

elif st.session_state.page == "AI Investigation":

    st.markdown(
        '<div class="page-title">AI Investigation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Evidence-first investigation using DealLens tools, '
        'RAG and Gemini.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="ai-panel">

            <div class="ai-label">
                DEAL LENS AI ANALYST
            </div>

            <h2>
                Investigate the target company.
            </h2>

            <p>
                Ask questions about financial performance,
                anomalies, dependencies, contracts or
                document evidence.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Investigation question",
        placeholder=(
            "Example: What are the most important "
            "financial and contractual risks that "
            "an acquirer should investigate?"
        ),
        height=120,
    )

    if st.button(
        "Run AI Investigation",
        type="primary",
        width="stretch",
    ):

        if not question.strip():

            st.warning(
                "Enter an investigation question first."
            )

        else:

            try:

                from agent.agent import investigate
                from agent.tools import Toolbox

                toolbox = Toolbox(
                    doc_index=pipeline.DOCS.get(
                        "index"
                    ),
                    state=pipeline.STATE,
                )

                with st.spinner(
                    "DealLens is investigating evidence..."
                ):

                    answer = investigate(
                        question,
                        toolbox,
                    )

                st.markdown(
                    """
                    <div class="section-card">

                        <div class="section-title">
                            Investigation Result
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(answer)

            except Exception as exc:

                st.error(
                    f"AI investigation failed: {exc}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DealLens AI • Evidence-first M&A intelligence • "
    "Deterministic analysis + ML signals + GenAI investigation"
)