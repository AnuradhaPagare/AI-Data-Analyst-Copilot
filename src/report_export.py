"""Export helpers: cleaned CSV and a plain-text analysis report."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from .analysis import compute_kpis, fmt_number, fmt_pct, numeric_summary

_LABELS = {"finding": "FINDING", "warning": "WARNING", "recommendation": "SUGGESTION", "hypothesis": "HYPOTHESIS (unverified)"}


def cleaned_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_report_text(filename: str, df: pd.DataFrame, roles: dict, quality: dict, insights: list[dict],
                      cleaning_log: list[str] | None = None, cleaned: bool = False) -> str:
    kpis = compute_kpis(df, roles)
    lines = [
        "AI DATA ANALYST COPILOT - ANALYSIS REPORT",
        "=" * 60,
        f"Dataset : {filename}",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"Data used: {'cleaned dataset' if cleaned else 'original dataset (not cleaned)'}",
        "",
        "1. DATASET OVERVIEW",
        "-" * 60,
        f"Rows: {quality['rows']:,} | Columns: {quality['columns']}",
        f"Missing cells: {quality['missing_total']:,} ({quality['missing_pct']:.2f}%)",
        f"Duplicate rows: {quality['duplicates']:,}",
        f"Potential outliers: " + (", ".join(f"{c} ({n})" for c, n in quality["outliers_by_column"].items()) or "none flagged"),
        "",
        "2. KEY METRICS",
        "-" * 60,
    ]
    labels = {"revenue": "Revenue", "profit": "Profit", "margin": "Profit margin", "orders": "Orders", "units": "Units sold",
              "avg_order_value": "Average order value", "records": "Records"}
    for k, v in kpis.items():
        lines.append(f"{labels[k]}: {fmt_pct(v) if k == 'margin' else fmt_number(v)}")
    summary = numeric_summary(df)
    if not summary.empty:
        lines += ["", "3. NUMERIC SUMMARY", "-" * 60]
        for col, r in summary.iterrows():
            lines.append(f"{col}: mean {fmt_number(r['mean'])}, median {fmt_number(r['median'])}, min {fmt_number(r['min'])}, max {fmt_number(r['max'])}")
    lines += ["", "4. INSIGHTS", "-" * 60]
    for i, ins in enumerate(insights, 1):
        lines.append(f"{i}. [{_LABELS.get(ins['kind'], ins['kind'].upper())}] {ins['title']}\n   {ins['detail']}")
    if cleaning_log:
        lines += ["", "5. CLEANING OPERATIONS PERFORMED", "-" * 60] + [f"- {l}" for l in cleaning_log]
    lines += ["", "LIMITATIONS", "-" * 60,
              "Findings are computed from the supplied data only. Correlations do not prove causation, and items marked HYPOTHESIS are unverified.", ""]
    return "\n".join(lines)
