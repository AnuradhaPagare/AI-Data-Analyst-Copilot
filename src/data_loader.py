"""Loading CSV / Excel files and generating the bundled sample dataset."""
from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
MAX_BYTES = 50 * 1024 * 1024  # 50 MB
SAMPLE_FILENAME = "sample_sales.csv"


class DataLoadError(Exception):
    """Raised with a user-friendly message when a file cannot be loaded."""


def load_dataframe(source: str | Path | bytes | BinaryIO, filename: str | None = None) -> pd.DataFrame:
    """Load a CSV/XLSX file from a path, raw bytes or a file-like object."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = filename or path.name
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise DataLoadError(f"File not found: {path.name}") from exc
        except OSError as exc:
            raise DataLoadError(f"Could not read {path.name}: {exc}") from exc
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        data = source.read()
        filename = filename or getattr(source, "name", None)

    if not filename:
        raise DataLoadError("A filename is required to detect the file type.")
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file type '{ext or 'unknown'}'. Please upload a .csv or .xlsx file."
        )
    if not data:
        raise DataLoadError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise DataLoadError(f"File is too large (limit {MAX_BYTES // (1024 * 1024)} MB).")

    try:
        df = _read_csv(data) if ext == ".csv" else pd.read_excel(io.BytesIO(data), engine="openpyxl")
    except DataLoadError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The file contains no data.") from exc
    except Exception as exc:  # malformed content, corrupt workbook, etc.
        raise DataLoadError(f"Could not parse the {ext} file: {exc}") from exc

    df = _tidy(df)
    if df.shape[1] == 0 or df.shape[0] == 0:
        raise DataLoadError("The file has no usable rows or columns.")
    return df


def _read_csv(data: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.ParserError as exc:
            raise DataLoadError(f"Malformed CSV: {exc}") from exc
    raise DataLoadError(f"Could not decode the CSV file: {last_error}")


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully empty rows/columns and make column names unique, clean strings."""
    df = df.dropna(how="all").dropna(axis=1, how="all")
    names, seen = [], {}
    for col in df.columns:
        name = str(col).strip() or "column"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        names.append(name)
    df = df.copy()
    df.columns = names
    return df.reset_index(drop=True)


def dataset_info(df: pd.DataFrame, filename: str) -> dict:
    return {
        "filename": filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
    }


# --------------------------------------------------------------------------- #
# Sample data
# --------------------------------------------------------------------------- #
_PRODUCTS = {
    # product: (category, list price, typical margin)
    "Laptop Pro 14": ("Electronics", 950, 0.18),
    "Smartphone X": ("Electronics", 620, 0.20),
    "Wireless Headphones": ("Electronics", 120, 0.30),
    "Tablet Air": ("Electronics", 380, 0.19),
    "Ergonomic Chair": ("Furniture", 210, 0.30),
    "Standing Desk": ("Furniture", 480, 0.27),
    "Bookshelf": ("Furniture", 150, 0.32),
    "Notebook Pack": ("Office Supplies", 18, 0.38),
    "Printer Paper": ("Office Supplies", 32, 0.30),
    "Desk Organizer": ("Office Supplies", 45, 0.40),
    "Ink Cartridge": ("Office Supplies", 65, 0.33),
    "Productivity Suite": ("Software", 140, 0.62),
    "Antivirus License": ("Software", 55, 0.66),
}


def generate_sample_sales(n_orders: int = 360, seed: int = 42) -> pd.DataFrame:
    """Create a realistic, internally consistent sales dataset.

    Includes a few intentional missing values, duplicates and bulk-order outliers,
    plus a late-2024 Electronics discount that lowers profit (a story to explore).
    """
    rng = np.random.default_rng(seed)
    products = list(_PRODUCTS)
    popularity = rng.dirichlet(np.ones(len(products)) * 4)
    # Orders skew slightly towards later dates (business growth)
    day_offsets = np.sort((rng.beta(1.3, 1.0, n_orders) * 730).astype(int))
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(day_offsets, unit="D")

    rows = []
    for i, date in enumerate(dates, start=1):
        product = rng.choice(products, p=popularity)
        category, list_price, margin = _PRODUCTS[product]
        qty = int(rng.integers(1, 6 if list_price > 200 else 12))
        if rng.random() < 0.01:  # rare bulk order -> outlier
            qty *= 12
        seasonal = 1.15 if date.month in (11, 12) else 1.0
        qty = max(1, int(round(qty * seasonal)))
        discount = 0.88 if (category == "Electronics" and date >= pd.Timestamp("2024-07-01")) else 1.0
        unit_price = round(list_price * rng.uniform(0.95, 1.05) * discount, 2)
        unit_cost = list_price * (1 - margin) * rng.uniform(0.97, 1.03)
        revenue = round(qty * unit_price, 2)
        cost = round(qty * unit_cost, 2)
        rows.append(
            {
                "order_id": f"ORD-{i:05d}",
                "order_date": date.strftime("%Y-%m-%d"),
                "product": product,
                "category": category,
                "region": rng.choice(["North", "South", "East", "West"], p=[0.28, 0.2, 0.27, 0.25]),
                "quantity": qty,
                "unit_price": unit_price,
                "revenue": revenue,
                "cost": cost,
                "profit": round(revenue - cost, 2),
                "customer_segment": rng.choice(["Consumer", "Corporate", "Small Business"], p=[0.45, 0.3, 0.25]),
            }
        )
    df = pd.DataFrame(rows)

    # Intentional missing values (demonstrates cleaning)
    for col, k in (("region", 6), ("customer_segment", 9), ("quantity", 3), ("unit_price", 3)):
        idx = rng.choice(df.index, size=k, replace=False)
        df[col] = df[col].astype("object") if col in ("region", "customer_segment") else df[col].astype(float)
        df.loc[idx, col] = np.nan
    # Intentional duplicate records
    dup = df.sample(7, random_state=seed)
    return pd.concat([df, dup], ignore_index=True).sort_values("order_id", kind="stable").reset_index(drop=True)


def ensure_sample_dataset(path: str | Path) -> Path:
    """Create the sample CSV if it does not exist yet and return its path."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_sample_sales().to_csv(path, index=False)
    return path
