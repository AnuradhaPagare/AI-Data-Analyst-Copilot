import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ai_analyst  # noqa: E402
from src.analysis import (  # noqa: E402
    apply_filters, compute_kpis, detect_columns, generate_insights, group_performance, numeric_summary,
)
from src.data_cleaning import CleaningOptions, clean_dataframe, coerce_date_columns, quality_report  # noqa: E402
from src.data_loader import DataLoadError, ensure_sample_dataset, generate_sample_sales, load_dataframe  # noqa: E402


@pytest.fixture
def small_df():
    return pd.DataFrame({
        "order_id": ["A1", "A2", "A3", "A3", "A4"],
        "order_date": ["2024-01-05", "2024-01-20", "2024-02-11", "2024-02-11", "2024-03-02"],
        "category": ["X", "Y", "X", "X", None],
        "quantity": [1, 2, 3, 3, np.nan],
        "revenue": [100.0, 200.0, 300.0, 300.0, 50.0],
        "profit": [10.0, 50.0, 60.0, 60.0, 5.0],
    })


# ---- loading -------------------------------------------------------------- #
def test_load_csv(tmp_path, small_df):
    p = tmp_path / "d.csv"
    small_df.to_csv(p, index=False)
    df = load_dataframe(p)
    assert df.shape == small_df.shape


def test_load_excel(tmp_path, small_df):
    p = tmp_path / "d.xlsx"
    small_df.to_excel(p, index=False)
    df = load_dataframe(p)
    assert df.shape == small_df.shape and "revenue" in df.columns


def test_load_from_bytes():
    df = load_dataframe(b"a,b\n1,2\n3,4\n", "x.csv")
    assert list(df.columns) == ["a", "b"] and len(df) == 2


def test_load_semicolon_csv():
    assert load_dataframe(b"a;b\n1;2\n3;4\n", "x.csv").shape == (2, 2)


@pytest.mark.parametrize("data,name", [(b"", "x.csv"), (b"a,b\n", "x.csv"), (b"junk", "x.txt"), (b"not an excel file", "x.xlsx")])
def test_load_errors_are_friendly(data, name):
    with pytest.raises(DataLoadError):
        load_dataframe(data, name)


def test_missing_file():
    with pytest.raises(DataLoadError):
        load_dataframe("does_not_exist.csv")


def test_sample_dataset_created_and_valid(tmp_path):
    p = ensure_sample_dataset(tmp_path / "data" / "sample_sales.csv")
    df = load_dataframe(p)
    assert len(df) >= 200
    for col in ["order_id", "order_date", "product", "category", "region", "quantity", "unit_price", "revenue", "cost", "profit", "customer_segment"]:
        assert col in df.columns
    q = quality_report(df)
    assert q["missing_total"] > 0 and q["duplicates"] > 0


def test_sample_is_internally_consistent():
    df = generate_sample_sales().dropna(subset=["quantity", "unit_price"])
    assert np.allclose(df["revenue"], (df["quantity"] * df["unit_price"]).round(2), atol=0.06 * df["quantity"].max())
    assert np.allclose(df["profit"], df["revenue"] - df["cost"], atol=0.02)


# ---- quality / cleaning --------------------------------------------------- #
def test_missing_and_duplicate_detection(small_df):
    q = quality_report(small_df)
    assert q["missing_total"] == 2
    assert q["missing_by_column"] == {"category": 1, "quantity": 1}
    assert q["duplicates"] == 1


def test_cleaning_does_not_mutate_original(small_df):
    original = small_df.copy(deep=True)
    res = clean_dataframe(small_df)
    pd.testing.assert_frame_equal(small_df, original)
    assert res.data is not small_df
    assert len(res.data) == 4 and res.after["missing_total"] == 0 and res.after["duplicates"] == 0
    assert any("duplicate" in l.lower() for l in res.log)


def test_cleaning_strategies(small_df):
    res = clean_dataframe(small_df, CleaningOptions(numeric_strategy="drop_rows", categorical_strategy="unknown", drop_duplicates=False))
    assert len(res.data) == 4 and (res.data["category"] == "Unknown").sum() == 0  # dropped row held the NaN category
    res2 = clean_dataframe(small_df, CleaningOptions(numeric_strategy="leave", categorical_strategy="leave"))
    assert res2.after["missing_total"] == 2


def test_median_and_mode_imputation(small_df):
    res = clean_dataframe(small_df, CleaningOptions(drop_duplicates=False))
    assert res.data.loc[4, "quantity"] == pd.Series([1, 2, 3, 3]).median()
    assert res.data.loc[4, "category"] == "X"


def test_date_parsing(small_df):
    out, converted = coerce_date_columns(small_df)
    assert converted == ["order_date"] and pd.api.types.is_datetime64_any_dtype(out["order_date"])
    assert not pd.api.types.is_datetime64_any_dtype(small_df["order_date"])  # original untouched


def test_empty_dataframe_handling():
    empty = pd.DataFrame()
    q = quality_report(empty)
    assert q["rows"] == 0 and q["missing_total"] == 0
    res = clean_dataframe(empty)
    assert res.data.empty
    assert ai_analyst.ask("what are the insights?", empty, detect_columns(empty))["text"]
    assert generate_insights(empty, detect_columns(empty))[0]["kind"] == "warning"


# ---- analysis ------------------------------------------------------------- #
def test_numeric_summary(small_df):
    s = numeric_summary(small_df)
    assert s.loc["revenue", "mean"] == pytest.approx(190.0)
    assert s.loc["quantity", "missing"] == 1


def test_kpis(small_df):
    df = clean_dataframe(small_df).data
    roles = detect_columns(df)
    k = compute_kpis(df, roles)
    assert k["revenue"] == pytest.approx(650.0)
    assert k["profit"] == pytest.approx(125.0)
    assert k["margin"] == pytest.approx(125 / 650)
    assert k["orders"] == 4
    assert k["avg_order_value"] == pytest.approx(650 / 4)


def test_kpis_absent_columns_are_not_invented():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    k = compute_kpis(df, detect_columns(df))
    assert set(k) == {"records"}


def test_group_performance_and_filters(small_df):
    df = clean_dataframe(small_df).data
    roles = detect_columns(df)
    perf = group_performance(df, "category", roles)
    assert perf.iloc[0]["category"] == "X" and perf.iloc[0]["revenue"] == pytest.approx(450.0)  # missing category was imputed with mode "X"
    filtered = apply_filters(df, "order_date", (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")), {"category": ["X"]})
    assert len(filtered) == 1


def test_sample_produces_five_insights_and_adapts():
    df = clean_dataframe(generate_sample_sales()).data
    roles = detect_columns(df)
    assert len(generate_insights(df, roles)) >= 5
    # a non-sales dataset must not get revenue/profit claims
    rng = np.random.default_rng(0)
    gen = pd.DataFrame({"temp": rng.normal(20, 5, 60), "humidity": rng.normal(50, 10, 60), "city": rng.choice(["A", "B", "C"], 60)})
    text = " ".join(i["title"] + i["detail"] for i in generate_insights(gen, detect_columns(gen))).lower()
    assert "revenue" not in text and "profit" not in text


# ---- AI analyst (no key) -------------------------------------------------- #
def test_chat_works_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    df = clean_dataframe(generate_sample_sales()).data
    roles = detect_columns(df)
    for q in ai_analyst.SUGGESTED_QUESTIONS + ["how many rows are there?", "average revenue by region", "any missing values?", "gibberish xyz"]:
        res = ai_analyst.ask(q, df, roles)
        assert res["source"] == "local" and res["text"] and "No API key" in res["warning"]
    assert "Electronics" in ai_analyst.ask("Which category performs best?", df, roles)["text"]


def test_llm_context_has_no_raw_rows():
    df = clean_dataframe(generate_sample_sales()).data
    roles = detect_columns(df)
    ctx = ai_analyst.build_llm_context("top category?", df, roles, ai_analyst.run_operations("top category?", df, roles))
    blob = str(ctx)
    assert "ORD-0000" not in blob  # no order ids / raw records


def test_api_failure_falls_back_to_local(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(ai_analyst, "_call_llm", lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    df = clean_dataframe(generate_sample_sales()).data
    res = ai_analyst.ask("Which category performs best?", df, detect_columns(df))
    assert res["source"] == "local" and "Electronics" in res["text"] and "sk-test" not in str(res["warning"])
