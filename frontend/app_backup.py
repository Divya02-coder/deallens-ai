"""
DealLens AI — recruiter-ready Streamlit investigation workspace.

Features:
- Deterministic financial analysis
- Risk register
- Transaction anomaly detection
- Customer/supplier dependency analysis
- NetworkX dependency graph
- Scenario simulation
- ChromaDB + SentenceTransformer RAG
- Gemini AI investigation agent
- Gemini executive report generation
"""

import os
import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dotenv import load_dotenv

from backend.services import pipeline

from finance_engine.scenarios import (
    Base,
    Scenario,
    run as run_scenario,
    sensitivity,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DealLens AI",
    page_icon="",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

SEV = {
    "High": "🔴",
    "Medium": "🟠",
    "Low": "🟡",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def cr(value):
    """Convert rupees to crore representation."""
    try:
        return f"₹{float(value) / 1e7:,.1f} Cr"
    except (TypeError, ValueError):
        return "₹0.0 Cr"


def safe_float(value, default=0.0):
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_percentage(value):
    """Format a numeric value as a percentage."""
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def clean_duplicate_columns(df):
    """
    Remove duplicate column names.

    This prevents Pandas from returning a Series when
    accessing something such as row['revenue_growth'].
    """
    if df is None:
        return df

    df = df.copy()

    duplicate_mask = df.columns.duplicated()

    if duplicate_mask.any():
        df = df.loc[:, ~duplicate_mask]

    return df


def latest_row(df):
    """
    Return the latest row after removing duplicate columns.
    """
    df = clean_duplicate_columns(df)

    if df is None or df.empty:
        return None

    if "year" in df.columns:
        df = df.sort_values("year")

    return df.iloc[-1]


# ============================================================
# OFFLINE INVESTIGATION
# ============================================================

def offline_investigation(question, state):
    """
    Evidence-backed fallback investigation assistant.

    Uses deterministic findings only.
    No API key is required.
    """

    q = question.lower()

    findings = state.get("findings", [])

    selected = findings

    # --------------------------------------------------------
    # Customer / concentration
    # --------------------------------------------------------

    if any(
        keyword in q
        for keyword in [
            "customer",
            "revenue concentration",
            "churn",
        ]
    ):
        selected = [
            finding
            for finding in findings
            if (
                finding.get("category") == "Dependency"
                and "Customer" in finding.get("title", "")
            )
        ]

    # --------------------------------------------------------
    # Supplier
    # --------------------------------------------------------

    elif any(
        keyword in q
        for keyword in [
            "supplier",
            "procurement",
        ]
    ):
        selected = [
            finding
            for finding in findings
            if "Supplier concentration"
            in finding.get("title", "")
        ]

    # --------------------------------------------------------
    # Financial / cash
    # --------------------------------------------------------

    elif any(
        keyword in q
        for keyword in [
            "cash",
            "profit",
            "earnings",
            "receivable",
            "dso",
            "margin",
        ]
    ):
        selected = [
            finding
            for finding in findings
            if finding.get("category")
            in [
                "Earnings quality",
                "Revenue quality",
                "Working capital",
            ]
        ]

    # --------------------------------------------------------
    # Transactions
    # --------------------------------------------------------

    elif any(
        keyword in q
        for keyword in [
            "transaction",
            "payment",
            "anomal",
            "threshold",
            "benford",
        ]
    ):
        selected = [
            finding
            for finding in findings
            if finding.get("category") == "Transactions"
        ]

    # --------------------------------------------------------
    # Debt / liquidity
    # --------------------------------------------------------

    elif any(
        keyword in q
        for keyword in [
            "debt",
            "liquidity",
            "leverage",
        ]
    ):
        selected = [
            finding
            for finding in findings
            if finding.get("category")
            in [
                "Liquidity",
                "Leverage",
            ]
        ]

    # --------------------------------------------------------
    # Contracts
    # --------------------------------------------------------

    elif any(
        keyword in q
        for keyword in [
            "contract",
            "renewal",
            "acquisition",
        ]
    ):
        selected = [
            finding
            for finding in findings
            if finding.get("category")
            == "Contract exposure"
        ]

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not selected:
        selected = findings[:5]

    lines = [
        "### Evidence-backed investigation",
        f"**Question:** {question}",
        "",
    ]

    for finding in selected[:5]:

        severity = finding.get(
            "severity",
            "Low",
        )

        title = finding.get(
            "title",
            "Unnamed finding",
        )

        detail = finding.get(
            "detail",
            "",
        )

        evidence = finding.get(
            "evidence",
            [],
        )

        questions = finding.get(
            "questions",
            [],
        )

        lines.extend(
            [
                (
                    f"**{SEV.get(severity, '🟡')} "
                    f"{title}**"
                ),
                detail,
                f"Evidence: `{evidence}`",
            ]
        )

        if questions:

            lines.append(
                "**Management questions:**"
            )

            for management_question in questions:
                lines.append(
                    f"- {management_question}"
                )

        lines.append("")

    lines.append(
        "_Offline mode uses deterministic project outputs. "
        "Enable the Gemini agent for natural-language "
        "multi-step investigation._"
    )

    return "\n".join(lines)


# ============================================================
# DEPENDENCY GRAPH
# ============================================================

def build_graph(
    company_name,
    customers,
    suppliers,
    debt,
    contracts,
):
    """
    Build a NetworkX dependency graph.

    The graph connects:
        Company
          ├── Customers
          ├── Suppliers
          ├── Lenders
          └── Contracts
    """

    graph = nx.Graph()

    graph.add_node(
        company_name,
        kind="company",
    )

    # --------------------------------------------------------
    # Customers
    # --------------------------------------------------------

    if isinstance(customers, pd.DataFrame):

        for _, row in customers.iterrows():

            customer = str(
                row.get(
                    "customer",
                    "Unknown customer",
                )
            )

            graph.add_node(
                customer,
                kind="customer",
            )

            graph.add_edge(
                company_name,
                customer,
                relationship="customer",
            )

    # --------------------------------------------------------
    # Suppliers
    # --------------------------------------------------------

    if isinstance(suppliers, pd.DataFrame):

        for _, row in suppliers.iterrows():

            supplier = str(
                row.get(
                    "supplier",
                    "Unknown supplier",
                )
            )

            graph.add_node(
                supplier,
                kind="supplier",
            )

            graph.add_edge(
                company_name,
                supplier,
                relationship="supplier",
            )

    # --------------------------------------------------------
    # Debt / lenders
    # --------------------------------------------------------

    if isinstance(debt, pd.DataFrame):

        lender_column = None

        for candidate in [
            "lender",
            "bank",
            "institution",
            "creditor",
        ]:
            if candidate in debt.columns:
                lender_column = candidate
                break

        if lender_column:

            for _, row in debt.iterrows():

                lender = str(
                    row.get(
                        lender_column,
                        "Unknown lender",
                    )
                )

                graph.add_node(
                    lender,
                    kind="lender",
                )

                graph.add_edge(
                    company_name,
                    lender,
                    relationship="debt",
                )

    # --------------------------------------------------------
    # Contracts
    # --------------------------------------------------------

    if isinstance(contracts, pd.DataFrame):

        contract_column = None

        for candidate in [
            "contract_id",
            "contract",
            "id",
        ]:
            if candidate in contracts.columns:
                contract_column = candidate
                break

        if contract_column:

            for _, row in contracts.iterrows():

                contract = str(
                    row.get(
                        contract_column,
                        "Unknown contract",
                    )
                )

                graph.add_node(
                    contract,
                    kind="contract",
                )

                graph.add_edge(
                    company_name,
                    contract,
                    relationship="contract",
                )

    return graph


def graph_figure(state):
    """
    Create interactive NetworkX + Plotly dependency graph.
    """

    graph = build_graph(
        "Target Co",
        state.get("cust"),
        state.get("sup"),
        state.get("debt"),
        state.get("contracts"),
    )

    if graph.number_of_nodes() == 0:

        return go.Figure()

    positions = nx.spring_layout(
        graph,
        seed=7,
        k=1.5,
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    edge_x = []
    edge_y = []

    for source, target in graph.edges():

        x0, y0 = positions[source]
        x1, y1 = positions[target]

        edge_x.extend(
            [
                x0,
                x1,
                None,
            ]
        )

        edge_y.extend(
            [
                y0,
                y1,
                None,
            ]
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1),
        hoverinfo="none",
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    node_x = []
    node_y = []
    hover_text = []
    node_sizes = []

    for node, data in graph.nodes(
        data=True
    ):

        x, y = positions[node]

        node_x.append(x)
        node_y.append(y)

        kind = data.get(
            "kind",
            "entity",
        )

        hover_text.append(
            f"{node}<br>Type: {kind}"
        )

        node_sizes.append(
            30
            if kind == "company"
            else 15
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(graph.nodes()),
        textposition="bottom center",
        hovertext=hover_text,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            line=dict(width=1),
        ),
    )

    fig = go.Figure(
        [
            edge_trace,
            node_trace,
        ]
    )

    fig.update_layout(
        title="Target-company dependency graph",
        showlegend=False,
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10,
        ),
        xaxis=dict(
            showgrid=False,
            visible=False,
        ),
        yaxis=dict(
            showgrid=False,
            visible=False,
        ),
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.title(" DealLens AI")

st.caption(
    "AI-powered M&A due-diligence & risk intelligence "
    "• evidence first • human analyst in the loop"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Investigation Workspace"
    )

    st.write(
        "Upload source documents, then run the "
        "deterministic analysis pipeline."
    )

    files = st.file_uploader(
        "Source documents",
        type=[
            "pdf",
            "csv",
            "xlsx",
        ],
        accept_multiple_files=True,
        help=(
            "PDFs are indexed for page-cited search. "
            "CSV/XLSX files can replace matching files "
            "in data/raw."
        ),
    )

    if files:

        for uploaded_file in files:

            destination = (
                PROJECT_ROOT
                / "data"
                / "raw"
                / Path(
                    uploaded_file.name
                ).name
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination.write_bytes(
                uploaded_file.getbuffer()
            )

            if (
                destination
                .suffix
                .lower()
                == ".pdf"
            ):

                try:

                    chunk_count = (
                        pipeline.add_pdf(
                            str(destination)
                        )
                    )

                    st.success(
                        f"Indexed "
                        f"{uploaded_file.name}: "
                        f"{chunk_count} chunks"
                    )

                except Exception as exc:

                    st.error(
                        f"PDF indexing failed: "
                        f"{exc}"
                    )

            else:

                st.success(
                    f"Loaded "
                    f"{uploaded_file.name}"
                )

    if st.button(
        "Run full analysis",
        type="primary",
        width="stretch",
    ):

        with st.spinner(
            "Running finance, anomaly, "
            "dependency and contract engines..."
        ):

            try:

                pipeline.analyze(
                    str(
                        PROJECT_ROOT
                        / "data"
                        / "raw"
                    )
                )

                st.success(
                    "Analysis complete."
                )

            except Exception as exc:

                st.error(
                    f"Analysis failed: {exc}"
                )

    st.divider()

    st.caption(
        "Numbers are calculated by Python. "
        "LLM interpretation is optional."
    )


# ============================================================
# PIPELINE STATE
# ============================================================

STATE = pipeline.STATE


if not STATE:

    st.info(
        "Click **Run full analysis** to analyze "
        "the included synthetic target company."
    )

    st.code(
        "python data/generate_sample_data.py\n"
        "python -m streamlit run frontend/app.py",
        language="bash",
    )

    st.stop()


# ============================================================
# TOP KPI STRIP
# ============================================================

findings = STATE.get(
    "findings",
    [],
)

high = sum(
    finding.get("severity") == "High"
    for finding in findings
)

medium = sum(
    finding.get("severity") == "Medium"
    for finding in findings
)


# ------------------------------------------------------------
# Clean ratios BEFORE selecting latest row
# ------------------------------------------------------------

ratios_df = clean_duplicate_columns(
    STATE.get("ratios")
)

latest = latest_row(
    ratios_df
)


# ------------------------------------------------------------
# Customer / supplier
# ------------------------------------------------------------

customers_df = STATE.get(
    "cust",
    pd.DataFrame(),
)

suppliers_df = STATE.get(
    "sup",
    pd.DataFrame(),
)

top_customer = None
top_supplier = None

if (
    isinstance(customers_df, pd.DataFrame)
    and not customers_df.empty
):

    customers_df = customers_df.copy()

    if "revenue" in customers_df.columns:

        top_customer = (
            customers_df
            .sort_values(
                "revenue",
                ascending=False,
            )
            .iloc[0]
        )


if (
    isinstance(suppliers_df, pd.DataFrame)
    and not suppliers_df.empty
):

    suppliers_df = suppliers_df.copy()

    if "procurement" in suppliers_df.columns:

        top_supplier = (
            suppliers_df
            .sort_values(
                "procurement",
                ascending=False,
            )
            .iloc[0]
        )


# ============================================================
# KPI DISPLAY
# ============================================================

kpi_columns = st.columns(5)


# ------------------------------------------------------------
# Risks
# ------------------------------------------------------------

kpi_columns[0].metric(
    "Risks requiring review",
    len(findings),
    f"{high} high",
)


# ------------------------------------------------------------
# Revenue growth
# ------------------------------------------------------------

revenue_growth = 0.0

if (
    latest is not None
    and "revenue_growth" in latest.index
):

    revenue_growth = safe_float(
        latest["revenue_growth"]
    )

kpi_columns[1].metric(
    "Revenue growth",
    safe_percentage(
        revenue_growth
    ),
)


# ------------------------------------------------------------
# EBITDA margin
# ------------------------------------------------------------

ebitda_margin = 0.0

if (
    latest is not None
    and "ebitda_margin" in latest.index
):

    ebitda_margin = safe_float(
        latest["ebitda_margin"]
    )

kpi_columns[2].metric(
    "EBITDA margin",
    safe_percentage(
        ebitda_margin
    ),
)


# ------------------------------------------------------------
# Top customer
# ------------------------------------------------------------

customer_concentration = 0.0

if (
    top_customer is not None
    and "revenue" in top_customer.index
):

    total_customer_revenue = safe_float(
        customers_df["revenue"].sum()
    )

    if total_customer_revenue:

        customer_concentration = (
            safe_float(
                top_customer["revenue"]
            )
            / total_customer_revenue
        )

kpi_columns[3].metric(
    "Top customer",
    safe_percentage(
        customer_concentration
    ),
)


# ------------------------------------------------------------
# Top supplier
# ------------------------------------------------------------

supplier_concentration = 0.0

if (
    top_supplier is not None
    and "procurement" in top_supplier.index
):

    total_procurement = safe_float(
        suppliers_df["procurement"].sum()
    )

    if total_procurement:

        supplier_concentration = (
            safe_float(
                top_supplier["procurement"]
            )
            / total_procurement
        )

kpi_columns[4].metric(
    "Top supplier",
    safe_percentage(
        supplier_concentration
    ),
)


# ============================================================
# MAIN TABS
# ============================================================

tabs = st.tabs(
    [
        "Executive View",
        "Risk Register",
        "Financials",
        "Dependencies",
        "Transactions",
        "Scenario",
        "Documents / AI",
    ]
)


# ============================================================
# EXECUTIVE VIEW
# ============================================================

with tabs[0]:

    st.subheader(
        "Executive investigation brief"
    )

    left, right = st.columns(
        [1.4, 1]
    )

    with left:

        st.markdown(
            "**Highest-priority themes**"
        )

        for finding in findings[:6]:

            severity = finding.get(
                "severity",
                "Low",
            )

            title = finding.get(
                "title",
                "Unnamed finding",
            )

            detail = finding.get(
                "detail",
                "",
            )

            st.markdown(
                f"{SEV.get(severity, '🟡')} "
                f"**{title}**  \n"
                f"{detail}"
            )

    with right:

        severity_df = pd.DataFrame(
            {
                "Severity": [
                    "High",
                    "Medium",
                    "Low",
                ],
                "Count": [
                    sum(
                        x.get("severity")
                        == "High"
                        for x in findings
                    ),
                    sum(
                        x.get("severity")
                        == "Medium"
                        for x in findings
                    ),
                    sum(
                        x.get("severity")
                        == "Low"
                        for x in findings
                    ),
                ],
            }
        )

        st.plotly_chart(
            px.bar(
                severity_df,
                x="Severity",
                y="Count",
                title="Finding severity",
                text="Count",
            ),
            width="stretch",
        )

    st.info(
        "This is intentionally not a single opaque "
        "'deal score'. DealLens separates deterministic "
        "calculations, ML screening signals and "
        "investigation questions."
    )

    # --------------------------------------------------------
    # AI Report
    # --------------------------------------------------------

    st.markdown(
        "### Generative AI Report"
    )

    if st.button(
        "Generate AI Due-Diligence Report",
        width="stretch",
    ):

        if os.getenv(
            "GEMINI_API_KEY"
        ):

            try:

                from ai_engine.report_generator import (
                    AIReportGenerator,
                )

                with st.spinner(
                    "Generating executive report..."
                ):

                    ai_report = (
                        AIReportGenerator()
                        .generate(STATE)
                    )

                st.session_state[
                    "ai_report"
                ] = ai_report

            except Exception as exc:

                st.error(
                    f"Report generation failed: "
                    f"{exc}"
                )

        else:

            st.warning(
                "GEMINI_API_KEY is not configured."
            )

    if st.session_state.get(
        "ai_report"
    ):

        st.markdown(
            st.session_state[
                "ai_report"
            ]
        )

        st.download_button(
            "⬇ Download AI report",
            st.session_state[
                "ai_report"
            ],
            "deallens_ai_report.md",
            "text/markdown",
            width="stretch",
        )


# ============================================================
# RISK REGISTER
# ============================================================

with tabs[1]:

    st.subheader(
        "Risk register"
    )

    for finding in findings:

        severity = finding.get(
            "severity",
            "Low",
        )

        rank = finding.get(
            "rank",
            "",
        )

        title = finding.get(
            "title",
            "Unnamed finding",
        )

        with st.expander(
            f"{SEV.get(severity, '🟡')} "
            f"#{rank} • {title}"
        ):

            left, right = st.columns(
                [1.6, 1]
            )

            with left:

                st.write(
                    finding.get(
                        "detail",
                        "",
                    )
                )

                st.markdown(
                    "**Evidence**"
                )

                st.json(
                    finding.get(
                        "evidence",
                        [],
                    )
                )

            with right:

                st.markdown(
                    "**Questions for management**"
                )

                for question in finding.get(
                    "questions",
                    [],
                ):

                    st.write(
                        "• " + str(question)
                    )

    st.caption(
        f"{len(findings)} findings generated from "
        "financials, transactions, customer/supplier "
        "data, debt and contracts."
    )


# ============================================================
# FINANCIALS
# ============================================================

with tabs[2]:

    st.subheader(
        "Deterministic financial analysis"
    )

    ratios = clean_duplicate_columns(
        STATE.get("ratios")
    )

    if (
        ratios is None
        or ratios.empty
    ):

        st.warning(
            "No financial ratio data available."
        )

    else:

        available_metrics = [
            metric_name
            for metric_name in [
                "revenue_growth",
                "ebitda_margin",
                "cash_conversion",
                "dso",
                "debt_to_ebitda",
                "interest_coverage",
            ]
            if metric_name
            in ratios.columns
        ]

        if available_metrics:

            metric = st.selectbox(
                "Trend metric",
                available_metrics,
            )

            st.plotly_chart(
                px.line(
                    ratios,
                    x="year",
                    y=metric,
                    markers=True,
                    title=(
                        metric
                        .replace(
                            "_",
                            " ",
                        )
                        .title()
                    ),
                ),
                width="stretch",
            )

        display_ratios = ratios.copy()

        if "year" in display_ratios.columns:

            display_ratios = (
                display_ratios
                .set_index("year")
                .T
            )

        st.dataframe(
            display_ratios,
            width="stretch",
        )

        st.caption(
            "All ratios are calculated in Python "
            "rather than generated by the LLM."
        )


# ============================================================
# DEPENDENCIES
# ============================================================

with tabs[3]:

    st.subheader(
        "Customer & supplier dependency intelligence"
    )

    left, right = st.columns(2)

    with left:

        if (
            isinstance(
                customers_df,
                pd.DataFrame,
            )
            and not customers_df.empty
            and "customer"
            in customers_df.columns
            and "revenue"
            in customers_df.columns
        ):

            st.plotly_chart(
                px.pie(
                    customers_df,
                    names="customer",
                    values="revenue",
                    hole=0.45,
                    title="Revenue concentration",
                ),
                width="stretch",
            )

        else:

            st.info(
                "Customer data unavailable."
            )

    with right:

        if (
            isinstance(
                suppliers_df,
                pd.DataFrame,
            )
            and not suppliers_df.empty
            and "supplier"
            in suppliers_df.columns
            and "procurement"
            in suppliers_df.columns
        ):

            st.plotly_chart(
                px.pie(
                    suppliers_df,
                    names="supplier",
                    values="procurement",
                    hole=0.45,
                    title="Procurement concentration",
                ),
                width="stretch",
            )

        else:

            st.info(
                "Supplier data unavailable."
            )

    st.plotly_chart(
        graph_figure(STATE),
        width="stretch",
    )

    st.caption(
        "The graph connects the target to customers, "
        "suppliers, lenders and contracts; this supports "
        "cross-document reasoning."
    )


# ============================================================
# TRANSACTIONS
# ============================================================

with tabs[4]:

    st.subheader(
        "Transaction anomaly detective"
    )

    scored_tx = STATE.get(
        "scored_tx"
    )

    benford = STATE.get(
        "benford",
        {},
    )

    if (
        not isinstance(
            scored_tx,
            pd.DataFrame,
        )
        or scored_tx.empty
    ):

        st.warning(
            "No transaction data available."
        )

    else:

        column_a, column_b, column_c = (
            st.columns(3)
        )

        # ----------------------------------------------------
        # Transaction count
        # ----------------------------------------------------

        column_a.metric(
            "Transactions screened",
            f"{len(scored_tx):,}",
        )

        # ----------------------------------------------------
        # Isolation Forest
        # ----------------------------------------------------

        outlier_count = 0

        if "is_outlier" in scored_tx.columns:

            outlier_count = int(
                scored_tx[
                    "is_outlier"
                ].sum()
            )

        column_a.metric(
            "Isolation Forest outliers",
            f"{outlier_count:,}",
        )

        # ----------------------------------------------------
        # Benford
        # ----------------------------------------------------

        chi_square = safe_float(
            benford.get(
                "chi_square",
                0,
            )
        )

        column_b.metric(
            "Benford χ²",
            f"{chi_square:.1f}",
        )

        column_b.metric(
            "Benford screening flag",
            (
                "Review"
                if benford.get(
                    "suspicious",
                    False,
                )
                else "No flag"
            ),
        )

        # ----------------------------------------------------
        # Top anomaly
        # ----------------------------------------------------

        top_transaction = (
            scored_tx.iloc[0]
        )

        top_vendor = str(
            top_transaction.get(
                "vendor",
                "N/A",
            )
        )

        anomaly_score = safe_float(
            top_transaction.get(
                "anomaly_score",
                0,
            )
        )

        column_c.metric(
            "Top anomaly vendor",
            top_vendor,
        )

        column_c.metric(
            "Highest anomaly score",
            f"{anomaly_score:.3f}",
        )

        # ----------------------------------------------------
        # Top transactions
        # ----------------------------------------------------

        top_transactions = (
            scored_tx
            .head(30)
            .copy()
        )

        if (
            "date" in top_transactions.columns
            and "amount"
            in top_transactions.columns
            and "anomaly_score"
            in top_transactions.columns
        ):

            chart_kwargs = {
                "x": "date",
                "y": "amount",
                "size": "anomaly_score",
                "title": (
                    "Highest-scoring transactions"
                ),
            }

            if "vendor" in top_transactions.columns:

                chart_kwargs[
                    "color"
                ] = "vendor"

            st.plotly_chart(
                px.scatter(
                    top_transactions,
                    **chart_kwargs,
                ),
                width="stretch",
            )

        display_columns = [
            column
            for column in [
                "date",
                "vendor",
                "amount",
                "anomaly_score",
                "is_outlier",
            ]
            if column
            in top_transactions.columns
        ]

        st.dataframe(
            top_transactions[
                display_columns
            ],
            width="stretch",
        )


# ============================================================
# SCENARIO SIMULATOR
# ============================================================

with tabs[5]:

    st.subheader(
        "M&A scenario simulator"
    )

    financials = STATE.get(
        "fin"
    )

    debt_df = STATE.get(
        "debt",
        pd.DataFrame(),
    )

    if (
        not isinstance(
            financials,
            pd.DataFrame,
        )
        or financials.empty
    ):

        st.warning(
            "Financial data is unavailable "
            "for scenario simulation."
        )

    else:

        latest_financials = (
            financials
            .sort_values("year")
            .iloc[-1]
        )

        # ----------------------------------------------------
        # Base model
        # ----------------------------------------------------

        total_debt = 0.0

        if (
            "total_debt"
            in latest_financials.index
        ):

            total_debt = safe_float(
                latest_financials[
                    "total_debt"
                ]
            )

        debt_principal = 0.0

        if (
            isinstance(
                debt_df,
                pd.DataFrame,
            )
            and "principal"
            in debt_df.columns
        ):

            debt_principal = safe_float(
                debt_df[
                    "principal"
                ].sum()
            )

        base = Base(
            safe_float(
                latest_financials.get(
                    "revenue",
                    0,
                )
            ),
            safe_float(
                latest_financials.get(
                    "cogs",
                    0,
                )
            ),
            safe_float(
                latest_financials.get(
                    "operating_expenses",
                    0,
                )
            ),
            safe_float(
                latest_financials.get(
                    "depreciation",
                    0,
                )
            ),
            safe_float(
                latest_financials.get(
                    "interest_expense",
                    0,
                )
            ),
            safe_float(
                latest_financials.get(
                    "capex",
                    0,
                )
            ),
            total_debt,
            safe_float(
                latest_financials.get(
                    "interest_expense",
                    0,
                )
            )
            + debt_principal * 0.2,
        )

        # ----------------------------------------------------
        # Scenario controls
        # ----------------------------------------------------

        revenue_change = (
            st.slider(
                "Revenue change %",
                -40,
                40,
                -15,
            )
            / 100
        )

        operating_cost_change = (
            st.slider(
                "Operating cost change %",
                -20,
                30,
                8,
            )
            / 100
        )

        lost_customer_revenue = (
            st.slider(
                "Lost customer revenue (Cr)",
                0,
                100,
                0,
            )
            * 1e7
        )

        annual_synergy = (
            st.slider(
                "Annual synergy (Cr)",
                0,
                100,
                0,
            )
            * 1e7
        )

        scenario = Scenario(
            revenue_change=revenue_change,
            opex_change=operating_cost_change,
            lost_customer_revenue=(
                lost_customer_revenue
            ),
            synergy=annual_synergy,
        )

        output = run_scenario(
            base,
            scenario,
        )

        # ----------------------------------------------------
        # Scenario KPIs
        # ----------------------------------------------------

        metrics = st.columns(4)

        metrics[0].metric(
            "Revenue",
            cr(
                output.get(
                    "revenue",
                    0,
                )
            ),
            cr(
                output.get(
                    "delta_revenue",
                    0,
                )
            ),
        )

        metrics[1].metric(
            "EBITDA",
            cr(
                output.get(
                    "ebitda",
                    0,
                )
            ),
            cr(
                output.get(
                    "delta_ebitda",
                    0,
                )
            ),
        )

        metrics[2].metric(
            "Free cash flow",
            cr(
                output.get(
                    "free_cash_flow",
                    0,
                )
            ),
            cr(
                output.get(
                    "delta_fcf",
                    0,
                )
            ),
        )

        metrics[3].metric(
            "Debt / EBITDA",
            (
                f"{safe_float(output.get('debt_to_ebitda')):.2f}x"
            ),
        )

        # ----------------------------------------------------
        # Sensitivity analysis
        # ----------------------------------------------------

        sensitivity_output = sensitivity(
            base,
            scenario,
            "revenue_change",
            [
                x / 100
                for x in range(
                    -30,
                    31,
                    5,
                )
            ],
        )

        sensitivity_df = pd.DataFrame(
            sensitivity_output
        )

        if not sensitivity_df.empty:

            st.plotly_chart(
                px.line(
                    sensitivity_df,
                    x="revenue_change",
                    y="ebitda",
                    markers=True,
                    title=(
                        "EBITDA sensitivity "
                        "to revenue growth"
                    ),
                ),
                width="stretch",
            )


# ============================================================
# DOCUMENTS / AI
# ============================================================

with tabs[6]:

    st.subheader(
        "Semantic RAG & AI Investigation"
    )

    st.caption(
        "Local embeddings + ChromaDB retrieval + "
        "optional Gemini reasoning."
    )

    # --------------------------------------------------------
    # Semantic RAG Search
    # --------------------------------------------------------

    document_query = st.text_input(
        "Semantic document search",
        placeholder=(
            "e.g. What contracts contain "
            "change-of-control clauses?"
        ),
    )

    if document_query:

        document_index = (
            pipeline.DOCS.get("index")
        )

        if document_index:

            try:

                hits = (
                    document_index
                    .search(
                        document_query,
                        k=5,
                    )
                )

                if hits:

                    for hit in hits:

                        source = hit.get(
                            "source",
                            "unknown",
                        )

                        page = hit.get(
                            "page",
                            0,
                        )

                        score = safe_float(
                            hit.get(
                                "score",
                                0,
                            )
                        )

                        with st.expander(
                            f"{source} • "
                            f"page {page} • "
                            f"relevance "
                            f"{score:.2f}"
                        ):

                            st.write(
                                hit.get(
                                    "text",
                                    "",
                                )
                            )

                else:

                    st.warning(
                        "No matching evidence found."
                    )

            except Exception as exc:

                st.error(
                    f"Document search failed: "
                    f"{exc}"
                )

        else:

            st.info(
                "Upload a PDF first to enable "
                "page-cited semantic document search."
            )

    st.divider()

    # --------------------------------------------------------
    # AI Investigation
    # --------------------------------------------------------

    question = st.text_input(
        "Investigation question",
        value=(
            "Find the top financial risks "
            "an acquirer should investigate."
        ),
    )

    if st.button(
        "Run AI Investigation",
        type="primary",
        width="stretch",
    ):

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if api_key:

            try:

                from agent.agent import (
                    investigate,
                )

                from agent.tools import (
                    Toolbox,
                )

                with st.spinner(
                    "DealLens AI is investigating "
                    "financials, documents and "
                    "dependencies..."
                ):

                    answer = investigate(
                        question,
                        Toolbox(
                            STATE,
                            pipeline.DOCS.get(
                                "index"
                            ),
                        ),
                    )

                st.markdown(
                    "###  AI Investigation"
                )

                st.markdown(
                    answer
                )

            except Exception as exc:

                st.error(
                    f"AI investigation failed: "
                    f"{exc}"
                )

                st.info(
                    "The deterministic DealLens "
                    "analysis is still available."
                )

                st.markdown(
                    offline_investigation(
                        question,
                        STATE,
                    )
                )

        else:

            st.warning(
                "GEMINI_API_KEY is not configured."
            )

            st.caption(
                "Add your free Gemini API key "
                "to .env to activate the AI agent."
            )

            st.markdown(
                offline_investigation(
                    question,
                    STATE,
                )
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DealLens AI • Evidence-first M&A due diligence "
    "• Deterministic finance + ML anomaly detection "
    "+ RAG + Gemini AI investigation"
)