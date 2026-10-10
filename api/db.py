import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.getenv("LOG_DB_PATH", "query_logs.db")


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT,
                best_score REAL,
                latency_sec REAL,
                status TEXT NOT NULL
            )
        """)


def log_query(question, answer, best_score, latency_sec, status):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO query_logs (created_at, question, answer, best_score, latency_sec, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), question, answer, best_score, latency_sec, status),
        )


def get_stats():
    with sqlite3.connect(DB_PATH) as con:
        total, errors, not_found, avg_latency = con.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN status = 'not_found' THEN 1 ELSE 0 END),
                   AVG(CASE WHEN status != 'error' THEN latency_sec END)
            FROM query_logs
        """).fetchone()
    return {
        "total_questions": total,
        "errors": errors or 0,
        "not_found": not_found or 0,
        "avg_latency_sec": round(avg_latency, 2) if avg_latency else None,
    }