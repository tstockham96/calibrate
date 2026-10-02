"""SQLite storage for decisions and outcomes."""
from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id   TEXT NOT NULL,      -- one per API call
    question_key  TEXT NOT NULL,
    ts            TEXT NOT NULL,      -- ISO-8601 UTC
    provider      TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'primary',  -- primary | shadow
    model         TEXT,               -- resolved model/version returned by the provider
    requested_model TEXT,
    question_hash TEXT NOT NULL,      -- hash of the question definition (instructions+criteria+type)
    state_hash    TEXT NOT NULL,      -- sha256 of canonical state; raw state is NOT stored by default
    qtype         TEXT NOT NULL,      -- noul | choice | score
    answer        TEXT,               -- choice label / score level / 'yes'|'no' at the logged threshold
    prob          REAL,               -- noul P(yes); choice/score top probability
    confidence    REAL,
    threshold     REAL,               -- noul threshold in force when the decision was made
    segment       TEXT,               -- free-form tag, e.g. channel=chat
    latency_ms    REAL,
    input_tokens  INTEGER,
    PRIMARY KEY (decision_id, question_key, role, provider)
);
CREATE TABLE IF NOT EXISTS outcomes (
    decision_id  TEXT NOT NULL,
    question_key TEXT NOT NULL,
    actual       TEXT NOT NULL,
    observed_at  TEXT,
    PRIMARY KEY (decision_id, question_key)
);
"""


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def insert_decision(con, row: dict):
    cols = ",".join(row.keys())
    qs = ",".join("?" for _ in row)
    con.execute(f"INSERT OR REPLACE INTO decisions ({cols}) VALUES ({qs})", list(row.values()))


def ingest_outcomes_csv(con, path: str) -> int:
    """CSV columns: decision_id, question_key, actual[, observed_at]. Returns rows ingested."""
    n = 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            con.execute("INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?)",
                        (r["decision_id"], r["question_key"], str(r["actual"]).strip(),
                         r.get("observed_at") or datetime.now(timezone.utc).isoformat()))
            n += 1
    con.commit()
    return n


def ingest_outcome(con, decision_id: str, question_key: str, actual, observed_at: str | None = None):
    """Programmatic/webhook-style single outcome."""
    con.execute("INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?)",
                (decision_id, question_key, str(actual),
                 observed_at or datetime.now(timezone.utc).isoformat()))
    con.commit()


def joined(con, question_key: str | None = None, role: str | None = "primary"):
    sql = ("SELECT d.*, o.actual FROM decisions d JOIN outcomes o "
           "ON d.decision_id=o.decision_id AND d.question_key=o.question_key WHERE 1=1")
    args = []
    if question_key:
        sql += " AND d.question_key=?"
        args.append(question_key)
    if role:
        sql += " AND d.role=?"
        args.append(role)
    return [dict(r) for r in con.execute(sql + " ORDER BY d.ts", args)]


def all_decisions(con, role: str | None = "primary"):
    sql = "SELECT * FROM decisions"
    args = []
    if role:
        sql += " WHERE role=?"
        args.append(role)
    return [dict(r) for r in con.execute(sql + " ORDER BY ts", args)]
