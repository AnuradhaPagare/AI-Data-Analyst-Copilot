"""Tiny optional SQLite store for chat history (local file, never contains the dataset)."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"


def _connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, ts TEXT, dataset TEXT, question TEXT, answer TEXT, source TEXT)")
    return conn


def save_entry(dataset: str, question: str, answer: str, source: str, path: Path = DB_PATH) -> None:
    try:
        with _connect(path) as conn:
            conn.execute("INSERT INTO history (ts, dataset, question, answer, source) VALUES (?,?,?,?,?)",
                         (datetime.now().isoformat(timespec="seconds"), dataset, question, answer, source))
    except sqlite3.Error:
        pass  # history is optional - never break the app


def recent(limit: int = 20, path: Path = DB_PATH) -> list[dict]:
    try:
        with _connect(path) as conn:
            rows = conn.execute("SELECT ts, dataset, question, source FROM history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"time": r[0], "dataset": r[1], "question": r[2], "source": r[3]} for r in rows]
    except sqlite3.Error:
        return []


def clear(path: Path = DB_PATH) -> None:
    try:
        with _connect(path) as conn:
            conn.execute("DELETE FROM history")
    except sqlite3.Error:
        pass
