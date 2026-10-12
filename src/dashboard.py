"""Streamlit/Plotly rendering helpers for the dashboard and EDA pages."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .analysis import (
    apply_filters, categorical_distribution, correlation_matrix, group_performance, has_business_metrics,
    kpi_cards, monthly_trend, primary_metric, strongest_correlations,
)

PALETTE = ["#7c3aed", "#db2777", "#06b6d4", "#f59e0b", "#10b981", "#3b82f6", "#f97316", "#64748b"]
_LAYOUT = dict(
    template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), colorway=PALETTE, height=380,
    legend=dict(orientation="h", y=-0.2), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#312e81"), title=dict(font=dict(size=16, color="#312e81")),
    xaxis=dict(gridcolor="#ece7ff"), yaxis=dict(gridcolor="#ece7ff"),
)


def show_plot(fig: go.Figure, key: str | None = None) -> None:
    fig.update_layout(**_LAYOUT)
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def render_kpis(df: pd.DataFrame, roles: dict) -> None:
    cards = kpi_cards(df, roles)
    for start in range(0, len(cards), 6):
        cols = st.columns(min(6, len(cards) - start))
        for col, (label, value, helptext) in zip(cols, cards[start:start + 6]):
            with col:
                st.metric(label, value, help=helptext)


def render_filters(df: pd.DataFrame, roles: dict, key_prefix: str) -> pd.DataFrame:
    """Date + categorical filters; returns the filtered frame."""
    date_col = roles.get("date")
    cats = [c for c in (roles.get("category"), roles.get("region"), roles.get("segment")) if c]
    if not cats:
        cats = roles.get("categorical", [])[:3]
    if not date_col and not cats:
        return df
    date_range, selected = None, {}
    with st.expander("Filters", expanded=True):
        widgets = ([date_col] if date_col else []) + cats
        cols = st.columns(len(widgets))
        i = 0
        if date_col:
            valid = df[date_col].dropna()
            if not valid.empty:
                lo, hi = valid.min().date(), valid.max().date()
                with cols[0]:
                    picked = st.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi, key=f"{key_prefix}_date")
                if isinstance(picked, (tuple, list)) and len(picked) == 2:
                    date_range = picked
            i = 1
        for col_widget, c in zip(cols[i:], cats):
            options = sorted(df[c].dropna().astype(str).unique())
            with col_widget:
                chosen = st.multiselect(c.replace("_", " ").title(), options, default=options, key=f"{key_prefix}_{c}")
            if len(chosen) != len(options):
                selected[c] = chosen
    work = df
    if selected:  # compare as strings so filter matches the multiselect options
        work = df.copy()
        mask = pd.Series(True, index=work.index)
        for c, chosen in selected.items():
            mask &= work[c].astype(str).isin(chosen)
        work = work[mask]
    return apply_filters(work, date_col, date_range)


def render_business_charts(df: pd.DataFrame, roles: dict) -> None:
    rev, prof, date_col = roles.get("revenue"), roles.get("profit"), roles.get("date")
    n_charts = 0
    if date_col:
        metrics = {}
        if rev:
            metrics["Revenue"] = (rev, "sum")
        if prof:
            metrics["Profit"] = (prof, "sum")
        if metrics:
            tr = monthly_trend(df, date_col, metrics)
            if len(tr) >= 2:
                fig = px.line(tr, x="month", y=list(metrics), markers=True, title="Monthly trend", labels={"value": "Amount", "month": "Month", "variable": ""})
                show_plot(fig, "dash_trend"); n_charts += 1
    cat, prod, reg, seg = (roles.get(k) for k in ("category", "product", "region", "segment"))
    left, right = st.columns(2)
    if cat and (rev or prof):
        perf = group_performance(df, cat, roles)
        with left:
            ycols = [c for c in ("revenue", "profit") if c in perf]
            fig = px.bar(perf, x=cat, y=ycols, barmode="group", title=f"Revenue vs profit by {cat}", labels={"value": "Amount", "variable": ""})
            show_plot(fig, "dash_cat"); n_charts += 1
        if "margin" in perf:
            with right:
                m = perf.sort_values("margin")
                fig = px.bar(m, x="margin", y=cat, orientation="h", title=f"Profit margin by {cat}", labels={"margin": "Profit margin"})
                fig.update_xaxes(tickformat=".0%")
                show_plot(fig, "dash_margin"); n_charts += 1
    left2, right2 = st.columns(2)
    if prod and (rev or prof):
        metric = "revenue" if rev else "profit"
        perf = group_performance(df, prod, roles).head(10).sort_values(metric)
        with left2:
            fig = px.bar(perf, x=metric, y=prod, orientation="h", title=f"Top {len(perf)} products by {metric}")
            show_plot(fig, "dash_prod"); n_charts += 1
    dim2 = reg or seg
    if dim2 and (rev or prof):
        metric = "revenue" if rev else "profit"
        perf = group_performance(df, dim2, roles)
        with right2:
            fig = px.pie(perf, names=dim2, values=metric, hole=0.45, title=f"{metric.title()} share by {dim2}")
            show_plot(fig, "dash_dim2"); n_charts += 1
    return None


def render_general_charts(df: pd.DataFrame, roles: dict) -> None:
    nums, cats, date_col = roles.get("numeric", []), roles.get("categorical", []), roles.get("date")
    st.info("No revenue/profit columns were detected, so a general-purpose dashboard is shown instead.")
    left, right = st.columns(2)
    if nums:
        with left:
            col = st.selectbox("Numeric column", nums, key="gen_num")
            show_plot(px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}"), "gen_hist")
    if cats:
        with right:
            col = st.selectbox("Categorical column", cats, key="gen_cat")
            dist = categorical_distribution(df, col, 12)
            show_plot(px.bar(dist, x=col, y="count", title=f"Most common values in {col}"), "gen_bar")
    if date_col:
        metrics = {"Records": (None, "count")}
        pm = primary_metric(df, roles)
        if pm:
            metrics[f"Mean {pm[0]}"] = (pm[0], "mean")
        tr = monthly_trend(df, date_col, metrics)
        if len(tr) >= 2:
            show_plot(px.line(tr, x="month", y="Records", markers=True, title="Records per month"), "gen_trend")
    corr = correlation_matrix(df)
    if not corr.empty:
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="PuOr_r", zmin=-1, zmax=1, title="Correlation matrix")
        show_plot(fig, "gen_corr")


def render_dashboard(df: pd.DataFrame, roles: dict, key_prefix: str) -> None:
    filtered = render_filters(df, roles, key_prefix)
    if filtered.empty:
        st.warning("No rows match the current filters. Widen the selection to see results.")
        return
    st.caption(f"Showing {len(filtered):,} of {len(df):,} rows")
    render_kpis(filtered, roles)
    st.divider()
    if has_business_metrics(roles):
        render_business_charts(filtered, roles)
    else:
        render_general_charts(filtered, roles)
    st.subheader("Filtered data preview")
    show_table(filtered.head(200), hide_index=True)


def render_correlations(df: pd.DataFrame, roles: dict) -> None:
    corr = correlation_matrix(df)
    if corr.empty:
        st.info("Correlations need at least two numeric columns with variation and 10+ rows.")
        return
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="PuOr_r", zmin=-1, zmax=1, title="Correlation matrix (Pearson)")
    show_plot(fig, "eda_corr")
    pairs = strongest_correlations(df, roles, min_abs=0.5)
    if pairs:
        tbl = pd.DataFrame([{"Column A": p["col_a"], "Column B": p["col_b"], "r": round(p["r"], 3),
                             "Note": "Arithmetic relationship (expected)" if p["mechanical"] else "Check for confounders"} for p in pairs])
        show_table(tbl, hide_index=True)
    st.caption("Correlation measures linear association only and does not imply causation.")
