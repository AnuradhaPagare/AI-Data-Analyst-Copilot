"""Column detection, EDA, KPI calculations and rule-based insight generation.

Everything here is deterministic Pandas/NumPy code: no AI involved. Business metrics
are only produced when the matching columns exist in the dataset.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .data_cleaning import is_text, iqr_bounds, quality_report

ROLE_PATTERNS: dict[str, list[str]] = {
    "revenue": ["revenue", "total_sales", "sales", "sales_amount", "total_amount", "amount", "turnover"],
    "profit": ["profit", "net_profit", "net_income"],
    "cost": ["cost", "cogs", "total_cost", "expense"],
    "quantity": ["quantity", "qty", "units", "units_sold"],
    "unit_price": ["unit_price", "price"],
    "order_id": ["order_id", "orderid", "order_no", "order_number", "invoice", "invoice_no", "transaction_id"],
    "product": ["product", "product_name", "item", "sku"],
    "category": ["category", "product_category", "department"],
    "region": ["region", "country", "state", "city", "area"],
    "segment": ["customer_segment", "segment"],
}
NUMERIC_ROLES = {"revenue", "profit", "cost", "quantity", "unit_price"}
MECHANICAL_ROLES = ("revenue", "profit", "cost", "quantity", "unit_price")
MAX_CATEGORY_CARDINALITY = 50


# --------------------------------------------------------------------------- #
# Column typing / detection
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _is_id_like(name: str) -> bool:
    n = _norm(name)
    return n == "id" or n.endswith("_id") or n.startswith("id_") or n.endswith("_no") or n.endswith("_number")


def numeric_columns(df: pd.DataFrame, include_ids: bool = False) -> list[str]:
    cols = [c for c in df.columns if ptypes.is_numeric_dtype(df[c]) and not ptypes.is_bool_dtype(df[c])]
    return cols if include_ids else [c for c in cols if not _is_id_like(c)]


def date_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if ptypes.is_datetime64_any_dtype(df[c])]


def categorical_columns(df: pd.DataFrame, max_cardinality: int = MAX_CATEGORY_CARDINALITY) -> list[str]:
    out = []
    for c in df.columns:
        s = df[c]
        if ptypes.is_datetime64_any_dtype(s) or ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s):
            continue
        if not (is_text(s) or ptypes.is_bool_dtype(s)) or _is_id_like(c):
            continue
        n_unique, n_valid = s.nunique(dropna=True), s.notna().sum()
        if 2 <= n_unique <= max_cardinality and (n_unique <= 0.5 * n_valid or n_valid < 20):
            out.append(c)
    return out


def _match_role(columns: dict[str, str], patterns: list[str], ok, used: set[str]) -> str | None:
    for p in patterns:  # exact normalised match first
        for col, norm in columns.items():
            if norm == p and col not in used and ok(col):
                return col
    for p in patterns:  # then whole-token match
        for col, norm in columns.items():
            if f"_{p}_" in f"_{norm}_" and col not in used and ok(col):
                return col
    return None


def detect_columns(df: pd.DataFrame) -> dict:
    """Detect semantic roles (revenue, profit, date, ...) and typed column lists."""
    columns = {c: _norm(c) for c in df.columns}
    used: set[str] = set()
    roles: dict = {}
    for role, patterns in ROLE_PATTERNS.items():
        if role in NUMERIC_ROLES:
            ok = lambda c: ptypes.is_numeric_dtype(df[c]) and not ptypes.is_bool_dtype(df[c])
        elif role == "order_id":
            ok = lambda c: True
        else:
            ok = lambda c: not ptypes.is_numeric_dtype(df[c]) and not ptypes.is_datetime64_any_dtype(df[c])
        col = _match_role(columns, patterns, ok, used)
        roles[role] = col
        if col:
            used.add(col)
    dates = date_columns(df)
    roles["date"] = next((c for c in dates if "date" in _norm(c)), dates[0] if dates else None)
    roles["dates"] = dates
    roles["numeric"] = numeric_columns(df)
    roles["categorical"] = categorical_columns(df)
    return roles


def has_business_metrics(roles: dict) -> bool:
    return bool(roles.get("revenue") or roles.get("profit"))


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_number(x, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:,.{decimals}f}B"
    if ax >= 1e6:
        return f"{x / 1e6:,.{decimals}f}M"
    if ax >= 1e4:
        return f"{x / 1e3:,.{decimals}f}K"
    if float(x).is_integer():
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def fmt_pct(x, decimals: int = 1) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x * 100:.{decimals}f}%"


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def apply_filters(df: pd.DataFrame, date_col: str | None = None, date_range=None, category_filters: dict | None = None) -> pd.DataFrame:
    """Filter by an inclusive date range and by selected values per categorical column."""
    out = df
    if date_col and date_range and len(date_range) == 2 and date_col in out:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        out = out[(out[date_col] >= start) & (out[date_col] <= end)]
    for col, selected in (category_filters or {}).items():
        if col in out and selected is not None:
            out = out[out[col].isin(list(selected))]
    return out


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = numeric_columns(df, include_ids=True)
    if not cols or df.empty:
        return pd.DataFrame()
    desc = df[cols].describe().T
    desc["missing"] = df[cols].isna().sum()
    desc["median"] = df[cols].median()
    desc["skew"] = df[cols].skew()
    ordered = ["count", "missing", "mean", "median", "std", "min", "25%", "75%", "max", "skew"]
    return desc[ordered]


def categorical_distribution(df: pd.DataFrame, col: str, top: int = 10) -> pd.DataFrame:
    counts = df[col].value_counts(dropna=False).head(top)
    out = counts.rename_axis(col).reset_index(name="count")
    out[col] = out[col].astype(object).where(out[col].notna(), "(missing)")
    out["share"] = out["count"] / len(df) if len(df) else 0.0
    return out


def correlation_matrix(df: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
    cols = [c for c in numeric_columns(df) if df[c].nunique(dropna=True) > 1]
    if len(cols) < 2 or len(df) < min_rows:
        return pd.DataFrame()
    return df[cols].corr(min_periods=min_rows)


def strongest_correlations(df: pd.DataFrame, roles: dict | None = None, min_abs: float = 0.5, top: int = 5) -> list[dict]:
    corr = correlation_matrix(df)
    if corr.empty:
        return []
    mech = {roles[r] for r in MECHANICAL_ROLES if roles and roles.get(r)} if roles else set()
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= min_abs:
                pairs.append({"col_a": a, "col_b": b, "r": float(r), "mechanical": a in mech and b in mech})
    return sorted(pairs, key=lambda p: -abs(p["r"]))[:top]


def outlier_details(df: pd.DataFrame, top: int = 5) -> list[dict]:
    out = []
    for col in numeric_columns(df, include_ids=False):
        b = iqr_bounds(df[col])
        if not b:
            continue
        x = df[col].dropna()
        mask = (x < b[0]) | (x > b[1])
        if mask.any():
            out.append({"column": col, "count": int(mask.sum()), "pct": float(mask.mean()), "lower": b[0], "upper": b[1],
                        "min": float(x.min()), "max": float(x.max())})
    return sorted(out, key=lambda d: -d["pct"])[:top]


# --------------------------------------------------------------------------- #
# KPIs and group performance
# --------------------------------------------------------------------------- #
def compute_kpis(df: pd.DataFrame, roles: dict) -> dict:
    """Return only the KPIs that the dataset's columns can support."""
    k: dict = {"records": int(len(df))}
    rev, prof, qty, oid = roles.get("revenue"), roles.get("profit"), roles.get("quantity"), roles.get("order_id")
    if rev:
        k["revenue"] = float(df[rev].sum())
    if prof:
        k["profit"] = float(df[prof].sum())
    if rev and prof and k["revenue"]:
        k["margin"] = k["profit"] / k["revenue"]
    if oid:
        k["orders"] = int(df[oid].nunique())
    if qty:
        k["units"] = float(df[qty].sum())
    if rev and oid and k.get("orders"):
        k["avg_order_value"] = k["revenue"] / k["orders"]
    return k


def kpi_cards(df: pd.DataFrame, roles: dict) -> list[tuple[str, str, str]]:
    """(label, formatted value, help text) tuples for the dashboard."""
    k = compute_kpis(df, roles)
    cards: list[tuple[str, str, str]] = []
    if "revenue" in k:
        cards.append(("Revenue", fmt_number(k["revenue"]), f"Sum of '{roles['revenue']}'"))
    if "profit" in k:
        cards.append(("Profit", fmt_number(k["profit"]), f"Sum of '{roles['profit']}'"))
    if "margin" in k:
        cards.append(("Profit margin", fmt_pct(k["margin"]), "Profit / revenue"))
    if "orders" in k:
        cards.append(("Orders", f"{k['orders']:,}", f"Unique '{roles['order_id']}' values"))
    if "units" in k:
        cards.append(("Units sold", fmt_number(k["units"]), f"Sum of '{roles['quantity']}'"))
    if "avg_order_value" in k:
        cards.append(("Avg order value", fmt_number(k["avg_order_value"]), "Revenue / unique orders"))
    if not cards:
        miss = quality_report(df)["missing_pct"] if len(df) else 0.0
        cards = [
            ("Records", f"{len(df):,}", "Rows in the current selection"),
            ("Columns", f"{df.shape[1]:,}", "Columns in the dataset"),
            ("Numeric columns", f"{len(roles.get('numeric', []))}", "Excluding ID-like columns"),
            ("Missing cells", f"{miss:.1f}%", "Share of empty cells"),
        ]
    return cards


def primary_metric(df: pd.DataFrame, roles: dict) -> tuple[str, str] | None:
    """The metric used for rankings: revenue (sum), else profit (sum), else first numeric (mean)."""
    if roles.get("revenue"):
        return roles["revenue"], "sum"
    if roles.get("profit"):
        return roles["profit"], "sum"
    nums = roles.get("numeric") or numeric_columns(df)
    return (nums[0], "mean") if nums else None


def group_performance(df: pd.DataFrame, by: str, roles: dict) -> pd.DataFrame:
    """Aggregate business metrics (or the primary numeric metric) per group."""
    if df.empty or by not in df:
        return pd.DataFrame()
    spec: dict[str, tuple[str, str]] = {}
    rev, prof, qty, oid = (roles.get(r) for r in ("revenue", "profit", "quantity", "order_id"))
    if rev:
        spec["revenue"] = (rev, "sum")
    if prof:
        spec["profit"] = (prof, "sum")
    if qty:
        spec["units"] = (qty, "sum")
    if oid:
        spec["orders"] = (oid, "nunique")
    if not (rev or prof):
        pm = primary_metric(df, roles)
        if pm:
            spec[f"{pm[1]}_{pm[0]}"] = pm
    spec["records"] = (by, "size")
    g = df.groupby(by, observed=True, dropna=True)
    out = g.agg(**{name: pd.NamedAgg(column=c, aggfunc=a) for name, (c, a) in spec.items()}).reset_index()
    if rev and prof:
        out["margin"] = np.where(out["revenue"] != 0, out["profit"] / out["revenue"].replace(0, np.nan), np.nan)
    sort_col = "revenue" if rev else "profit" if prof else [c for c in out.columns if c not in (by, "records")][0] if len(out.columns) > 2 else "records"
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
def monthly_trend(df: pd.DataFrame, date_col: str, metrics: dict[str, tuple[str | None, str]]) -> pd.DataFrame:
    """Monthly aggregation. ``metrics`` maps output name -> (column or None for row count, agg)."""
    d = df.dropna(subset=[date_col]).copy()
    if d.empty:
        return pd.DataFrame(columns=["month", *metrics])
    d["month"] = d[date_col].dt.to_period("M").dt.to_timestamp()
    parts = {}
    for name, (col, agg) in metrics.items():
        parts[name] = d.groupby("month").size() if col is None else d.groupby("month")[col].agg(agg)
    return pd.DataFrame(parts).sort_index().reset_index()


def drop_partial_last_month(trend: pd.DataFrame, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Drop the final month if the data stops before month end (avoids false 'declines')."""
    if len(trend) >= 3:
        last = df[date_col].max()
        if pd.notna(last) and not last.is_month_end:
            return trend.iloc[:-1]
    return trend


def change_drivers(df: pd.DataFrame, roles: dict, metric_col: str | None = None, window: int = 3) -> dict | None:
    """Compare the last ``window`` complete months with the previous ``window`` months."""
    date_col = roles.get("date")
    metric_col = metric_col or roles.get("profit") or roles.get("revenue")
    if not date_col or not metric_col or df.empty:
        return None
    trend = drop_partial_last_month(monthly_trend(df, date_col, {"v": (metric_col, "sum")}), df, date_col)
    if len(trend) < 2 * window:
        return None
    recent_m, prior_m = trend["month"].iloc[-window:], trend["month"].iloc[-2 * window:-window]
    month = df[date_col].dt.to_period("M").dt.to_timestamp()
    recent, prior = df[month.isin(recent_m)], df[month.isin(prior_m)]
    tot_recent, tot_prior = float(recent[metric_col].sum()), float(prior[metric_col].sum())
    out = {
        "metric": metric_col,
        "recent_period": (recent_m.min(), recent_m.max()),
        "prior_period": (prior_m.min(), prior_m.max()),
        "recent": tot_recent,
        "prior": tot_prior,
        "delta": tot_recent - tot_prior,
        "pct": (tot_recent - tot_prior) / abs(tot_prior) if tot_prior else np.nan,
        "by": {},
    }
    rev = roles.get("revenue")
    if rev and metric_col == roles.get("profit"):
        r_rev, p_rev = float(recent[rev].sum()), float(prior[rev].sum())
        out["margin_recent"] = tot_recent / r_rev if r_rev else np.nan
        out["margin_prior"] = tot_prior / p_rev if p_rev else np.nan
    for dim in (roles.get("category"), roles.get("product"), roles.get("region"), roles.get("segment")):
        if dim:
            a = prior.groupby(dim, observed=True)[metric_col].sum().rename("prior")
            b = recent.groupby(dim, observed=True)[metric_col].sum().rename("recent")
            t = pd.concat([a, b], axis=1).fillna(0.0)
            t["delta"] = t["recent"] - t["prior"]
            out["by"][dim] = t.sort_values("delta").reset_index()
    return out


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
def _ins(kind: str, title: str, detail: str) -> dict:
    """kind: finding | warning | recommendation | hypothesis"""
    return {"kind": kind, "title": title, "detail": detail}


def _dimension_insights(df: pd.DataFrame, roles: dict, dim: str, label: str) -> list[dict]:
    pm = primary_metric(df, roles)
    if not pm or df[dim].nunique() < 2:
        return []
    col, agg = pm
    perf = group_performance(df, dim, roles)
    if perf.empty or len(perf) < 2:
        return []
    out = []
    if agg == "sum":
        total = perf["revenue"].sum() if "revenue" in perf else perf["profit"].sum()
        key = "revenue" if "revenue" in perf else "profit"
        top, bottom = perf.iloc[0], perf.iloc[-1]
        share = top[key] / total if total else np.nan
        out.append(_ins("finding", f"Top {label} by {key}: {top[dim]}",
                        f"{top[dim]} generated {fmt_number(top[key])} ({fmt_pct(share)} of total {key}) across {len(perf)} {label} values; "
                        f"the lowest was {bottom[dim]} at {fmt_number(bottom[key])}."))
        if "margin" in perf and perf["margin"].notna().sum() >= 2:
            hi, lo = perf.loc[perf["margin"].idxmax()], perf.loc[perf["margin"].idxmin()]
            if hi[dim] != lo[dim]:
                out.append(_ins("finding", f"Profit margin differs by {label}",
                                f"{hi[dim]} has the highest margin ({fmt_pct(hi['margin'])}) and {lo[dim]} the lowest ({fmt_pct(lo['margin'])}). "
                                f"{lo[dim]} contributes {fmt_number(lo['revenue'])} revenue but only {fmt_number(lo['profit'])} profit."))
    else:
        name = f"{agg}_{col}"
        top, bottom = perf.iloc[0], perf.iloc[-1]
        out.append(_ins("finding", f"Highest average '{col}' by {label}: {top[dim]}",
                        f"{top[dim]} averages {fmt_number(top[name])} vs {fmt_number(bottom[name])} for {bottom[dim]} (lowest)."))
    return out


def generate_insights(df: pd.DataFrame, roles: dict, report: dict | None = None) -> list[dict]:
    """Produce evidence-based findings. Each item is labelled finding/warning/recommendation/hypothesis."""
    if df.empty:
        return [_ins("warning", "No data in the current selection", "Adjust filters or load a dataset.")]
    report = report or quality_report(df)
    out: list[dict] = []
    recs: list[dict] = []

    # 1. data quality
    if report["missing_total"] or report["duplicates"]:
        worst = sorted(report["missing_by_column"].items(), key=lambda kv: -kv[1])[:3]
        parts = []
        if report["missing_total"]:
            parts.append(f"{report['missing_total']:,} missing cells ({report['missing_pct']:.2f}% of all cells)"
                         + (", mostly in " + ", ".join(f"'{c}' ({n})" for c, n in worst) if worst else ""))
        if report["duplicates"]:
            parts.append(f"{report['duplicates']:,} fully duplicated rows")
        out.append(_ins("warning", "Data-quality issues detected", "; ".join(parts) + ". Results below may be affected until these are cleaned."))
        recs.append(_ins("recommendation", "Clean the data before reporting", "Run the Data Cleaning step, then confirm that duplicates were true duplicates and not legitimate repeat records."))
    else:
        out.append(_ins("finding", "No missing values or duplicate rows", f"All {report['rows']:,} rows are complete and unique."))

    # 2. dimensions
    dims = [(roles.get("category"), "category"), (roles.get("region"), "region"), (roles.get("segment"), "segment")]
    dims = [(c, l) for c, l in dims if c]
    if not dims:
        dims = [(c, c) for c in roles.get("categorical", [])[:2]]
    for col, label in dims[:3]:
        out.extend(_dimension_insights(df, roles, col, label))
    if roles.get("product") and roles.get("revenue") and df[roles["product"]].nunique() >= 4:
        perf = group_performance(df, roles["product"], roles)
        total = perf["revenue"].sum()
        if total:
            top3 = perf["revenue"].head(3).sum() / total
            out.append(_ins("finding", "Revenue concentration in top products",
                            f"The top 3 of {len(perf)} products ({', '.join(map(str, perf[roles['product']].head(3)))}) account for {fmt_pct(top3)} of revenue; "
                            f"the best seller is {perf.iloc[0][roles['product']]} ({fmt_pct(perf.iloc[0]['revenue'] / total)})."))
    if not has_business_metrics(roles):  # generic dataset: look at distributions
        for col in roles.get("categorical", [])[:2]:
            vc = df[col].value_counts(normalize=True)
            if len(vc) >= 2:
                out.append(_ins("finding", f"Most common value in '{col}'", f"'{vc.index[0]}' makes up {fmt_pct(vc.iloc[0])} of records ({len(vc)} distinct values)."))
        nums = roles.get("numeric", [])
        cv = {c: df[c].std() / abs(df[c].mean()) for c in nums if df[c].mean() not in (0, np.nan) and df[c].notna().sum() > 2 and pd.notna(df[c].std())}
        if cv:
            c = max(cv, key=cv.get)
            out.append(_ins("finding", f"Most variable numeric column: '{c}'", f"Its standard deviation is {cv[c] * 100:.0f}% of its mean (mean {fmt_number(df[c].mean())}, range {fmt_number(df[c].min())} to {fmt_number(df[c].max())})."))

    # 3. time trends
    date_col = roles.get("date")
    pm = primary_metric(df, roles)
    if date_col and df[date_col].notna().sum() > 0:
        col, agg = pm if pm else (None, "count")
        trend = monthly_trend(df, date_col, {"v": (col if pm else None, agg if pm else "count")})
        trend = drop_partial_last_month(trend, df, date_col)
        label = f"monthly {agg} of '{col}'" if pm else "monthly record count"
        if len(trend) >= 6:
            peak = trend.loc[trend["v"].idxmax()]
            low = trend.loc[trend["v"].idxmin()]
            last3, prev3 = trend["v"].iloc[-3:].mean(), trend["v"].iloc[-6:-3].mean()
            pct = (last3 - prev3) / abs(prev3) if prev3 else np.nan
            direction = "up" if pct > 0 else "down"
            half = len(trend) // 2
            h1, h2 = trend["v"].iloc[:half].mean(), trend["v"].iloc[half:].mean()
            out.append(_ins("finding", f"Trend in {label}",
                            f"The last 3 months averaged {fmt_number(last3)}, {direction} {fmt_pct(abs(pct))} vs the prior 3 months. "
                            f"Peak month: {peak['month']:%b %Y} ({fmt_number(peak['v'])}); lowest: {low['month']:%b %Y} ({fmt_number(low['v'])}). "
                            f"The second half of the period averaged {fmt_number(h2)} vs {fmt_number(h1)} in the first half."))
        elif len(trend) >= 2:
            out.append(_ins("finding", "Short time span", f"Only {len(trend)} complete month(s) of data - too few to establish a reliable trend."))

    # 4. what changed (profit/revenue drivers)
    drv = change_drivers(df, roles)
    if drv:
        word = "fell" if drv["delta"] < 0 else "rose"
        detail = (f"{drv['metric']} {word} {fmt_pct(abs(drv['pct']))} ({fmt_number(drv['delta'])}) in {drv['recent_period'][0]:%b %Y}-{drv['recent_period'][1]:%b %Y} "
                  f"versus {drv['prior_period'][0]:%b %Y}-{drv['prior_period'][1]:%b %Y}.")
        for dim, t in drv["by"].items():
            worst = t.iloc[0]
            if drv["delta"] < 0 and worst["delta"] < 0:
                detail += f" Largest decline by {dim}: {worst[dim]} ({fmt_number(worst['delta'])})."
                break
            if drv["delta"] >= 0:
                best = t.iloc[-1]
                detail += f" Largest gain by {dim}: {best[dim]} ({fmt_number(best['delta'])})."
                break
        if "margin_recent" in drv and pd.notna(drv["margin_recent"]) and pd.notna(drv["margin_prior"]):
            detail += f" Margin moved from {fmt_pct(drv['margin_prior'])} to {fmt_pct(drv['margin_recent'])}."
        out.append(_ins("finding", f"Recent change in {drv['metric']}", detail))
        if drv["delta"] < 0:
            dim = next(iter(drv["by"]), None)
            worst = drv["by"][dim].iloc[0][dim] if dim else None
            out.append(_ins("hypothesis", "Possible explanations (not proven by this data)",
                            "A drop concentrated in one segment can come from pricing or discounting, rising costs, lower volume or mix shifts. "
                            "The figures show where the change occurred, not why it occurred." ))
            if worst is not None:
                recs.append(_ins("recommendation", f"Investigate '{worst}' first",
                                 f"It accounts for the largest decline in {drv['metric']}. Compare unit price, unit cost and volume before and after the change."))

    # 5. outliers
    outs = outlier_details(df)
    if outs:
        o = outs[0]
        out.append(_ins("warning", "Potential outliers detected",
                        "; ".join(f"'{d['column']}': {d['count']} value(s) ({d['pct'] * 100:.1f}%) outside {fmt_number(d['lower'])}-{fmt_number(d['upper'])}" for d in outs[:3])
                        + f". Largest value in '{o['column']}' is {fmt_number(o['max'])}. These may be genuine bulk transactions or data-entry errors."))
        recs.append(_ins("recommendation", "Review the flagged outliers", "Verify the extreme records with the data owner before excluding them - they can distort averages and trends."))

    # 6. correlations (skip arithmetic relationships such as revenue = quantity x price)
    for p in [p for p in strongest_correlations(df, roles) if not p["mechanical"]][:1]:
        out.append(_ins("finding", f"Correlation between '{p['col_a']}' and '{p['col_b']}'",
                        f"Pearson r = {p['r']:.2f} across {len(df):,} rows. Correlation does not imply causation; check for confounders before acting on it."))

    return out + recs
