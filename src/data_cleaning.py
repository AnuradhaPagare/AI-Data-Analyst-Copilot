"""Data-quality reporting and configurable, non-destructive cleaning."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


def is_text(s: pd.Series) -> bool:
    return (
        ptypes.is_object_dtype(s)
        or ptypes.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ) and not ptypes.is_bool_dtype(s)


def _blank_mask(s: pd.Series) -> pd.Series:
    """True for NaN or blank strings."""
    mask = s.isna()
    if is_text(s) and not isinstance(s.dtype, pd.CategoricalDtype):
        mask = mask | s.astype("string").str.strip().eq("").fillna(False).astype(bool)
    return mask


def missing_counts(df: pd.DataFrame) -> pd.Series:
    return pd.Series({c: int(_blank_mask(df[c]).sum()) for c in df.columns}, dtype="int64")


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float] | None:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 8:
        return None
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return None
    return float(q1 - k * iqr), float(q3 + k * iqr)


def outlier_counts(df: pd.DataFrame) -> dict[str, int]:
    out = {}
    for col in df.select_dtypes(include="number").columns:
        if ptypes.is_bool_dtype(df[col]):
            continue
        bounds = iqr_bounds(df[col])
        if bounds:
            x = df[col].dropna()
            n = int(((x < bounds[0]) | (x > bounds[1])).sum())
            if n:
                out[col] = n
    return out


def quality_report(df: pd.DataFrame) -> dict:
    """Summarise missing values, duplicates, dtypes and IQR outliers."""
    rows, cols = df.shape
    miss = missing_counts(df) if cols else pd.Series(dtype="int64")
    miss_nonzero = miss[miss > 0]
    return {
        "rows": int(rows),
        "columns": int(cols),
        "missing_total": int(miss.sum()),
        "missing_by_column": {k: int(v) for k, v in miss_nonzero.items()},
        "missing_pct": float(miss.sum() / (rows * cols) * 100) if rows and cols else 0.0,
        "duplicates": int(df.duplicated().sum()) if rows else 0,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "outliers_by_column": outlier_counts(df) if rows else {},
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
    }


def coerce_date_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy where date-like text columns are converted to datetime."""
    out = df.copy()
    converted = []
    for col in out.columns:
        s = out[col]
        if ptypes.is_datetime64_any_dtype(s) or not is_text(s) or isinstance(s.dtype, pd.CategoricalDtype):
            continue
        non_null = s.dropna().astype(str).str.strip()
        non_null = non_null[non_null != ""]
        if len(non_null) < 3:
            continue
        name_hint = any(h in col.lower() for h in ("date", "time", "day", "month", "year", "period"))
        looks_dateish = non_null.head(50).str.contains(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|^\d{4}-\d{2}|:|^[A-Za-z]{3,9}\s+\d", regex=True).mean() >= 0.8
        if not (name_hint or looks_dateish):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s.astype("object").where(s.notna(), None), errors="coerce")
        if parsed.notna().sum() / len(non_null) >= 0.9:
            out[col] = parsed
            converted.append(col)
    return out, converted


@dataclass
class CleaningOptions:
    strip_whitespace: bool = True
    parse_dates: bool = True
    drop_duplicates: bool = True
    numeric_strategy: str = "median"  # median | mean | zero | drop_rows | leave
    categorical_strategy: str = "mode"  # mode | unknown | drop_rows | leave
    cap_outliers: bool = False  # winsorise numeric values to the IQR fences


@dataclass
class CleaningResult:
    data: pd.DataFrame
    log: list[str] = field(default_factory=list)
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)


def clean_dataframe(df: pd.DataFrame, options: CleaningOptions | None = None) -> CleaningResult:
    """Clean a copy of ``df`` according to ``options``; the input is never mutated."""
    options = options or CleaningOptions()
    work = df.copy(deep=True)
    log: list[str] = []
    before = quality_report(work)

    if work.empty:
        return CleaningResult(work, ["Dataset is empty - nothing to clean."], before, before)

    if options.strip_whitespace:
        n_blank = 0
        for col in work.columns:
            s = work[col]
            if is_text(s) and not isinstance(s.dtype, pd.CategoricalDtype):
                stripped = s.astype("string").str.strip()
                blanks = stripped.eq("").fillna(False).astype(bool)
                n_blank += int(blanks.sum())
                stripped = stripped.mask(blanks, pd.NA)
                work[col] = stripped.astype("object").where(stripped.notna(), np.nan)
        log.append(f"Trimmed whitespace in text columns; converted {n_blank} blank string(s) to missing values.")

    if options.parse_dates:
        work, converted = coerce_date_columns(work)
        if converted:
            log.append(f"Converted column(s) to datetime: {', '.join(converted)}.")

    if options.drop_duplicates:
        n = int(work.duplicated().sum())
        work = work.drop_duplicates().reset_index(drop=True)
        log.append(f"Removed {n} duplicate row(s)." if n else "No duplicate rows found.")
    else:
        log.append("Duplicate rows were kept (option disabled).")

    num_cols = [c for c in work.select_dtypes(include="number").columns if not ptypes.is_bool_dtype(work[c])]
    dt_cols = [c for c in work.columns if ptypes.is_datetime64_any_dtype(work[c])]
    cat_cols = [c for c in work.columns if c not in num_cols and c not in dt_cols]

    # --- numeric missing values
    num_missing = {c: int(work[c].isna().sum()) for c in num_cols if work[c].isna().any()}
    if num_missing:
        strat = options.numeric_strategy
        if strat in ("median", "mean", "zero"):
            for c in num_missing:
                fill = {"median": work[c].median(), "mean": work[c].mean(), "zero": 0}[strat]
                work[c] = work[c].fillna(fill)
                log.append(f"Filled {num_missing[c]} missing value(s) in numeric column '{c}' with the {strat if strat != 'zero' else 'constant 0'} ({fill:,.4g}).")
        elif strat == "drop_rows":
            n0 = len(work)
            work = work.dropna(subset=list(num_missing)).reset_index(drop=True)
            log.append(f"Dropped {n0 - len(work)} row(s) with missing numeric values.")
        else:
            log.append(f"Left {sum(num_missing.values())} missing numeric value(s) untouched.")

    # --- categorical missing values
    cat_missing = {c: int(work[c].isna().sum()) for c in cat_cols if work[c].isna().any()}
    if cat_missing:
        strat = options.categorical_strategy
        if strat == "mode":
            for c in cat_missing:
                modes = work[c].mode(dropna=True)
                if len(modes):
                    work[c] = work[c].fillna(modes.iloc[0])
                    log.append(f"Filled {cat_missing[c]} missing value(s) in '{c}' with the mode ('{modes.iloc[0]}').")
                else:
                    log.append(f"Column '{c}' has no mode (all missing) - left untouched.")
        elif strat == "unknown":
            for c in cat_missing:
                work[c] = work[c].fillna("Unknown")
                log.append(f"Filled {cat_missing[c]} missing value(s) in '{c}' with 'Unknown'.")
        elif strat == "drop_rows":
            n0 = len(work)
            work = work.dropna(subset=list(cat_missing)).reset_index(drop=True)
            log.append(f"Dropped {n0 - len(work)} row(s) with missing categorical values.")
        else:
            log.append(f"Left {sum(cat_missing.values())} missing categorical value(s) untouched.")

    dt_missing = {c: int(work[c].isna().sum()) for c in dt_cols if work[c].isna().any()}
    if dt_missing:
        log.append("Missing dates were left untouched (not imputed): " + ", ".join(f"{c} ({n})" for c, n in dt_missing.items()) + ".")

    # --- outliers
    if options.cap_outliers:
        capped = {}
        for c in num_cols:
            b = iqr_bounds(work[c])
            if b:
                n = int(((work[c] < b[0]) | (work[c] > b[1])).sum())
                if n:
                    work[c] = work[c].clip(b[0], b[1])
                    capped[c] = n
        log.append("Capped outliers to IQR fences: " + ", ".join(f"{c} ({n})" for c, n in capped.items()) + "." if capped else "No outliers needed capping.")
    else:
        n_out = sum(before["outliers_by_column"].values())
        log.append(f"Outliers were flagged but not modified ({n_out} potential outlier value(s) across {len(before['outliers_by_column'])} column(s)).")

    log.append(f"Result: {len(df):,} -> {len(work):,} rows; original upload was left unchanged.")
    return CleaningResult(work, log, before, quality_report(work))
