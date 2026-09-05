"""SQLite persistence (Section 13.1). WAL mode so a power loss corrupts at
most the last transaction -- the whole point of the offline-resilience
demo (Section 13.3): the pipeline never blocks on the dashboard or the
network, only on this local file.
"""
from __future__ import annotations

import pathlib
import sqlite3

_SCHEMA_PATH = pathlib.Path(__file__).parent / "schema.sql"


class Store:
    def __init__(self, path: str = "saresq.db"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA_PATH.read_text())
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _insert(self, table: str, **fields) -> int:
        cols = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        cur = self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", tuple(fields.values()))
        self.conn.commit()
        return cur.lastrowid

    @staticmethod
    def _unreserve(fields: dict) -> dict:
        """The schema's `class` column collides with the Python keyword;
        callers pass class_=... and it's translated back here."""
        if "class_" in fields:
            fields = dict(fields)
            fields["class"] = fields.pop("class_")
        return fields

    def insert_target(self, **fields) -> int:
        return self._insert("targets", **self._unreserve(fields))

    def update_target(self, target_id: int, **fields) -> None:
        fields = self._unreserve(fields)
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(f"UPDATE targets SET {sets} WHERE target_id = ?", (*fields.values(), target_id))
        self.conn.commit()

    def insert_pass(self, **fields) -> int:
        return self._insert("passes", **fields)

    def insert_hazard(self, **fields) -> int:
        return self._insert("hazards", **self._unreserve(fields))

    def insert_frame(self, **fields) -> int:
        return self._insert("frames", **fields)

    def get_target(self, target_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM targets WHERE target_id = ?", (target_id,)).fetchone()
        return dict(row) if row else None

    def get_passes_for_target(self, target_id: int) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM passes WHERE target_id = ? ORDER BY pass_id", (target_id,)).fetchall()
        return [dict(r) for r in rows]

    def all_targets(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM targets ORDER BY last_seen_ns DESC").fetchall()
        return [dict(r) for r in rows]

    def all_hazards(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM hazards ORDER BY t_ns DESC").fetchall()
        return [dict(r) for r in rows]
