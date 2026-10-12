"""AI Data Analyst Copilot - Streamlit app.  Run with:  streamlit run app.py"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from src import ai_analyst, history  # noqa: E402
from src.analysis import (  # noqa: E402
    categorical_distribution, detect_columns, generate_insights, numeric_summary,
)
from src.dashboard import render_correlations, render_dashboard, show_plot, show_table  # noqa: E402
from src.data_cleaning import CleaningOptions, clean_dataframe, coerce_date_columns, quality_report  # noqa: E402
from src.data_loader import DataLoadError, dataset_info, ensure_sample_dataset, load_dataframe  # noqa: E402
from src.report_export import build_report_text, cleaned_csv_bytes  # noqa: E402

import plotly.express as px  # noqa: E402

SAMPLE_PATH = BASE_DIR / "data" / "sample_sales.csv"
PAGES = ["Upload & Overview", "Data Cleaning", "Exploratory Analysis", "Dashboard", "Insights", "AI Analyst Chat", "Export"]

st.set_page_config(page_title="AI Data Analyst Copilot", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    /* page background: soft lavender -> pink -> sky gradient */
    [data-testid="stApp"] {background: linear-gradient(135deg, #ede9fe 0%, #fce7f3 50%, #e0f2fe 100%); background-attachment: fixed;}
    [data-testid="stHeader"] {background: transparent;}
    .block-container {padding-top: 2rem; max-width: 1300px;}
    /* gradient page title */
    h1 {background: linear-gradient(90deg, #7c3aed 0%, #db2777 100%); -webkit-background-clip: text;
        background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800 !important; letter-spacing: -0.5px;}
    h2, h3 {color: #312e81;}
    .app-sub {color: #6b7280; margin-top: -0.6rem; margin-bottom: 1.2rem;}
    /* sidebar */
    section[data-testid="stSidebar"] {background: rgba(255, 255, 255, 0.92); border-right: 1px solid #e9e1ff;}
    /* KPI cards */
    div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #ece7ff; border-left: 5px solid #7c3aed;
        border-radius: 12px; padding: 12px 16px; box-shadow: 0 2px 10px rgba(124, 58, 237, 0.08);}
    div[data-testid="stMetricValue"] {color: #4c1d95; font-weight: 700;}
    div[data-testid="stMetricLabel"] {color: #6b7280;}
    /* buttons */
    button[kind="primary"], button[data-testid="stBaseButton-primary"] {background: linear-gradient(90deg, #7c3aed, #db2777);
        border: none; color: #fff; font-weight: 600; border-radius: 10px;}
    button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {filter: brightness(1.08); box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);}
    button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {border-radius: 10px; border: 1px solid #d8ccff; color: #4c1d95;}
    button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {border-color: #7c3aed; background: #f3efff; color: #5b21b6;}
    /* containers, expanders, alerts */
    div[data-testid="stExpander"], div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px; background: rgba(255, 255, 255, 0.75);}
    div[data-testid="stAlert"] {border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {"raw_df": None, "filename": None, "source_id": None, "clean": None, "chat": [], "version": 0, "pending_q": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def set_dataset(df: pd.DataFrame, filename: str, source_id: str) -> None:
    st.session_state.update(raw_df=df, filename=filename, source_id=source_id, clean=None, chat=[])
    st.session_state.version += 1


def active_df() -> tuple[pd.DataFrame | None, bool]:
    """Cleaned data if cleaning was run, otherwise the original with date columns typed (original untouched)."""
    if st.session_state.clean is not None:
        return st.session_state.clean.data, True
    raw = st.session_state.raw_df
    if raw is None:
        return None, False
    return coerce_date_columns(raw)[0], False


def empty_state() -> None:
    st.info("**No dataset loaded yet.** Upload a CSV/XLSX file in the sidebar, or load the built-in sample sales dataset to explore the app.")
    if st.button("Load sample sales dataset", type="primary", key="empty_sample"):
        load_sample()
        st.rerun()


def load_sample() -> None:
    try:
        path = ensure_sample_dataset(SAMPLE_PATH)
        set_dataset(load_dataframe(path), path.name, "sample")
    except DataLoadError as exc:
        st.error(str(exc))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> str:
    st.sidebar.title("📊 Copilot")
    page = st.sidebar.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.subheader("Data source")
    up = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], key="uploader")
    if up is not None:
        sid = f"{up.name}-{up.size}"
        if sid != st.session_state.source_id:
            with st.sidebar:
                with st.spinner("Reading file..."):
                    try:
                        df = load_dataframe(up.getvalue(), up.name)
                        set_dataset(df, up.name, sid)
                        st.success(f"Loaded {up.name}")
                    except DataLoadError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Unexpected error while loading the file: {type(exc).__name__}")
    if st.sidebar.button("Load sample sales dataset", key="side_sample"):
        load_sample()
        st.rerun()
    if st.session_state.raw_df is not None:
        info = dataset_info(st.session_state.raw_df, st.session_state.filename)
        st.sidebar.caption(f"**{info['filename']}** · {info['rows']:,} rows · {info['columns']} cols")
        st.sidebar.caption("Analysing: **cleaned data**" if st.session_state.clean is not None else "Analysing: **original data** (not cleaned)")
    st.sidebar.divider()
    if ai_analyst.ai_available():
        st.sidebar.success(f"AI explanations: ON ({ai_analyst.get_model()})")
    else:
        st.sidebar.info("AI explanations: OFF - using local rule-based analysis. Add an API key in `.env` to enable.")
    return page


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    raw = st.session_state.raw_df
    if raw is None:
        empty_state()
        return
    info = dataset_info(raw, st.session_state.filename)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File", info["filename"])
    c2.metric("Rows", f"{info['rows']:,}")
    c3.metric("Columns", f"{info['columns']}")
    c4.metric("Memory", f"{info['memory_mb']:.2f} MB")
    st.subheader("Data preview (original upload)")
    show_table(raw.head(100), hide_index=True)
    st.subheader("Column types")
    types = pd.DataFrame({"column": raw.columns, "type": [str(t) for t in raw.dtypes], "non-null": raw.notna().sum().values, "unique": [raw[c].nunique() for c in raw.columns]})
    show_table(types, hide_index=True)
    st.caption("Next: open **Data Cleaning** to fix missing values and duplicates, or jump to the **Dashboard**.")


def _quality_block(report: dict, title: str) -> None:
    st.markdown(f"**{title}**")
    a, b, c = st.columns(3)
    a.metric("Missing cells", f"{report['missing_total']:,}", f"{report['missing_pct']:.2f}%", delta_color="off")
    b.metric("Duplicate rows", f"{report['duplicates']:,}")
    c.metric("Columns with outliers", f"{len(report['outliers_by_column'])}")
    if report["missing_by_column"]:
        show_table(pd.DataFrame({"column": report["missing_by_column"].keys(), "missing": report["missing_by_column"].values()}), hide_index=True)
    if report["outliers_by_column"]:
        st.caption("Potential outliers (IQR rule): " + ", ".join(f"{c} ({n})" for c, n in report["outliers_by_column"].items()))


def page_cleaning() -> None:
    raw = st.session_state.raw_df
    if raw is None:
        empty_state()
        return
    st.write("Cleaning is applied to a **copy**; your original upload is never modified.")
    with st.expander("Cleaning options", expanded=True):
        c1, c2 = st.columns(2)
        num = c1.selectbox("Missing numeric values", ["median", "mean", "zero", "drop_rows", "leave"], format_func=lambda x: {"median": "Fill with median", "mean": "Fill with mean", "zero": "Fill with 0", "drop_rows": "Drop rows", "leave": "Leave as is"}[x])
        cat = c2.selectbox("Missing categorical values", ["mode", "unknown", "drop_rows", "leave"], format_func=lambda x: {"mode": "Fill with most common value (mode)", "unknown": "Fill with 'Unknown'", "drop_rows": "Drop rows", "leave": "Leave as is"}[x])
        c3, c4, c5, c6 = st.columns(4)
        dup = c3.checkbox("Remove duplicate rows", value=True)
        strip = c4.checkbox("Trim whitespace", value=True)
        dates = c5.checkbox("Parse date columns", value=True)
        cap = c6.checkbox("Cap outliers (IQR)", value=False, help="Clips extreme numeric values to the 1.5xIQR fences. Off by default.")
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Run cleaning", type="primary"):
        with st.spinner("Cleaning data..."):
            res = clean_dataframe(raw, CleaningOptions(strip_whitespace=strip, parse_dates=dates, drop_duplicates=dup, numeric_strategy=num, categorical_strategy=cat, cap_outliers=cap))
        st.session_state.clean = res
        st.session_state.version += 1
        st.success("Cleaning complete. All analysis pages now use the cleaned data.")
    if b2.button("Reset to original"):
        st.session_state.clean = None
        st.session_state.version += 1
        st.info("Reverted to the original data.")

    res = st.session_state.clean
    before = res.before if res else quality_report(raw)
    left, right = st.columns(2)
    with left:
        _quality_block(before, "Before cleaning")
    with right:
        if res:
            _quality_block(res.after, "After cleaning")
        else:
            st.info("Run cleaning to see the after-cleaning report.")
    if res:
        st.subheader("Operations performed")
        for line in res.log:
            st.markdown(f"- {line}")
        st.subheader("Cleaned data preview")
        show_table(res.data.head(100), hide_index=True)


def page_eda() -> None:
    df, _ = active_df()
    if df is None:
        empty_state()
        return
    roles = detect_columns(df)
    q = quality_report(df)
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{q['rows']:,}"); b.metric("Columns", q["columns"]); c.metric("Missing cells", f"{q['missing_total']:,}"); d.metric("Duplicate rows", f"{q['duplicates']:,}")
    st.subheader("Numeric statistics")
    s = numeric_summary(df)
    if s.empty:
        st.info("No numeric columns found.")
    else:
        show_table(s.round(3))
    st.subheader("Categorical distributions")
    cats = roles["categorical"]
    if cats:
        col = st.selectbox("Column", cats, key="eda_cat")
        dist = categorical_distribution(df, col, 15)
        fig = px.bar(dist, x=col, y="count", text="count", title=f"Distribution of {col}")
        show_plot(fig, "eda_dist")
    else:
        st.info("No low-cardinality categorical columns found.")
    st.subheader("Correlations")
    render_correlations(df, roles)
    st.subheader("Date-based trends")
    if roles["date"] and roles["numeric"]:
        from src.analysis import monthly_trend
        c1, c2 = st.columns(2)
        metric = c1.selectbox("Metric", roles["numeric"], key="eda_metric")
        agg = c2.selectbox("Aggregation", ["sum", "mean", "median"], key="eda_agg")
        tr = monthly_trend(df, roles["date"], {metric: (metric, agg)})
        if len(tr) >= 2:
            show_plot(px.line(tr, x="month", y=metric, markers=True, title=f"Monthly {agg} of {metric}"), "eda_trend")
        else:
            st.info("Not enough distinct months for a trend.")
    else:
        st.info("No date column was detected (or no numeric columns), so date trends are unavailable. If dates are stored as text, run **Data Cleaning** with 'Parse date columns'.")


def page_dashboard() -> None:
    df, _ = active_df()
    if df is None:
        empty_state()
        return
    roles = detect_columns(df)
    render_dashboard(df, roles, key_prefix=f"dash{st.session_state.version}")


def page_insights() -> None:
    df, cleaned = active_df()
    if df is None:
        empty_state()
        return
    roles = detect_columns(df)
    ins = generate_insights(df, roles)
    if not cleaned:
        st.warning("Insights are based on the **original, uncleaned** data. Run Data Cleaning for more reliable results.")
    icons = {"finding": "🔎 Calculated finding", "warning": "⚠️ Warning", "recommendation": "✅ Suggestion", "hypothesis": "💭 Hypothesis (unverified)"}
    st.caption(f"{len(ins)} data-driven items. Findings are calculated from your data; hypotheses are possible explanations and are not proven.")
    for item in ins:
        with st.container(border=True):
            st.markdown(f"**{item['title']}**  \n<small>{icons[item['kind']]}</small>", unsafe_allow_html=True)
            st.write(item["detail"])


def page_chat() -> None:
    df, _ = active_df()
    if df is None:
        empty_state()
        return
    roles = detect_columns(df)
    if ai_analyst.ai_available():
        st.caption("AI explanations are on: your question, the schema and **aggregated computed results** are sent to the model - never the raw rows.")
    else:
        st.info("No API key found - answers come from local, rule-based analysis computed with Pandas. No AI model is called.")
    st.write("**Try a question:**")
    cols = st.columns(3)
    for i, q in enumerate(ai_analyst.SUGGESTED_QUESTIONS):
        if cols[i % 3].button(q, key=f"sugg_{i}"):
            st.session_state.pending_q = q
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m["role"] == "assistant":
                badge = f"🤖 AI explanation ({m.get('model', '')})" if m.get("source") == "ai" else "🧮 Local analysis (no AI model used)"
                st.caption(badge)
                if m.get("warning"):
                    st.warning(m["warning"])
                if m.get("sections"):
                    with st.expander("Computed results used"):
                        st.json([s for s in m["sections"]], expanded=False)
    prompt = st.chat_input("Ask a question about your data...") or st.session_state.pop("pending_q", None)
    if prompt:
        st.session_state.chat.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                res = ai_analyst.ask(prompt, df, roles)
            st.markdown(res["text"])
            st.caption(f"🤖 AI explanation ({res.get('model', '')})" if res["source"] == "ai" else "🧮 Local analysis (no AI model used)")
            if res.get("warning"):
                st.warning(res["warning"])
        st.session_state.chat.append({"role": "assistant", "text": res["text"], "source": res["source"], "model": res.get("model"), "warning": res.get("warning"), "sections": res.get("sections")})
        history.save_entry(st.session_state.filename or "", prompt, res["text"], res["source"])
        st.rerun()
    with st.expander("Recent questions (local history)"):
        rows = history.recent(10)
        st.write("\n".join(f"- {r['time']} - {r['question']} ({r['source']})" for r in rows) if rows else "No history yet.")
        if rows and st.button("Clear history"):
            history.clear()
            st.rerun()


def page_export() -> None:
    df, cleaned = active_df()
    if df is None:
        empty_state()
        return
    roles = detect_columns(df)
    report = quality_report(df)
    ins = generate_insights(df, roles, report)
    log = st.session_state.clean.log if st.session_state.clean else None
    base = Path(st.session_state.filename or "dataset").stem
    if not cleaned:
        st.warning("You haven't run Data Cleaning, so the CSV below is the original data (dates parsed only for analysis).")
    st.subheader("Cleaned dataset" if cleaned else "Dataset")
    st.download_button("⬇️ Download dataset as CSV", cleaned_csv_bytes(st.session_state.clean.data if cleaned else st.session_state.raw_df), file_name=f"{base}_cleaned.csv" if cleaned else f"{base}.csv", mime="text/csv")
    st.subheader("Analysis report")
    text = build_report_text(st.session_state.filename or "dataset", df, roles, report, ins, log, cleaned)
    st.download_button("⬇️ Download analysis report (TXT)", text.encode("utf-8"), file_name=f"{base}_report.txt", mime="text/plain", type="primary")
    with st.expander("Preview report"):
        st.code(text, language=None)


def main() -> None:
    init_state()
    page = sidebar()
    st.title("AI Data Analyst Copilot")
    st.markdown('<p class="app-sub">Upload data, clean it, explore it and ask questions - calculations run locally with Pandas.</p>', unsafe_allow_html=True)
    {"Upload & Overview": page_overview, "Data Cleaning": page_cleaning, "Exploratory Analysis": page_eda, "Dashboard": page_dashboard,
     "Insights": page_insights, "AI Analyst Chat": page_chat, "Export": page_export}[page]()


main()
