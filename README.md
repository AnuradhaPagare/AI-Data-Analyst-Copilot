# AI Data Analyst Copilot

A Streamlit app that lets you upload a CSV/Excel file, clean it, explore it, view an adaptive dashboard, and ask natural-language questions about it. All numbers are computed locally with Pandas; an LLM is **optional** and only explains aggregated results.

## Features
- **Upload** CSV/XLSX (or load the built-in sample); shows filename, rows, columns, memory; graceful errors for empty/unsupported/malformed files
- **Cleaning** (on a copy, original never overwritten): missing values, duplicates, dtypes, IQR outliers, before/after report, exact log of operations; median/mean/mode/"Unknown"/drop options
- **EDA**: numeric stats, categorical distributions, correlations, monthly trends
- **Dashboard**: KPI cards, date + category/region/segment filters, monthly trend, revenue vs profit, margins, top products, filtered preview. Adapts to the columns found; non-sales data gets a general-purpose dashboard and no invented business metrics
- **Insights**: 5+ evidence-based findings, warnings, suggestions, clearly labelled hypotheses
- **AI Analyst chat**: predefined, safe analytical operations; works fully without an API key
- **Export**: cleaned CSV + TXT analysis report

## Technology stack
Python 3.11+, Streamlit, Pandas, NumPy, Plotly, OpenPyXL, SQLite (chat history), python-dotenv, Anthropic API (optional).

## Architecture
```
app.py                 Streamlit UI (7 pages, session state)
src/data_loader.py     CSV/XLSX loading, validation, sample-data generator
src/data_cleaning.py   quality report, date parsing, configurable cleaning
src/analysis.py        column-role detection, EDA, KPIs, trends, insight engine
src/dashboard.py       filters + Plotly charts
src/ai_analyst.py      intent router -> predefined Pandas operations -> optional LLM explanation
src/history.py         optional SQLite chat history (data/history.db)
src/report_export.py   CSV / TXT export
tests/                 pytest suite
```
**AI safety design:** the model never writes or runs code or SQL. A keyword router selects from a fixed set of Pandas operations; only the question, the schema and the *aggregated results* are sent to the model (never raw rows). Answers separate "Calculated" findings from "Possible explanations (hypotheses)".

## Windows setup (Command Prompt)
```
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```
(macOS/Linux: `source venv/bin/activate` instead of `venv\Scripts\activate`.)

## Using the sample dataset
Click **Load sample sales dataset** in the sidebar. `data/sample_sales.csv` is regenerated automatically if missing. It has ~360 orders (2023-2024) with intentional missing values, 7 duplicate rows and a few bulk-order outliers.

## Enabling AI insights
1. Copy `.env.example` to `.env` (`copy .env.example .env` on Windows).
2. Set `ANTHROPIC_API_KEY=...` (optionally `ANTHROPIC_MODEL=...`).
3. Restart the app. The sidebar shows "AI explanations: ON". Each chat answer is badged as *AI explanation* or *Local analysis*, so you always know whether a model was called. On a missing key or API error the app falls back to local analysis and says so. `.env` is git-ignored; the key is never shown in the UI.

## Testing
```
python -m pytest -q
```

## Known limitations
- The question router is keyword-based; unusual phrasing falls back to an overview and a list of supported questions.
- Date detection works on columns whose values look like dates; ambiguous formats (e.g. 03/04/2024) follow pandas' default parsing.
- Outliers use the 1.5xIQR rule, which flags legitimate skewed values (e.g. big orders) - they are flagged, never removed by default.
- Pearson correlation only captures linear association; "Why" answers show *where* a change happened, not its cause.
- Designed for datasets that fit in memory (50 MB upload limit).
- Streamlit's file-upload and chat UI were verified via automated app tests, not a manual browser session.

## Future enhancements
Multi-file joins, saved dashboards, forecasting, more chart types, column-type overrides, LLM tool-use over a whitelisted operation set, PDF reports.
