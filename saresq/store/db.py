"""SQLite persistence (Section 13.1). WAL mode so a power loss corrupts at
most the last transaction -- the whole point of the offline-resilience
demo (Section 13.3): the pipeline never blocks on the dashboard or the
network, only on this local file.
"""
from __future__ import annotations

import pathlib
import sqlite3

_SCHEMA_PATH = pathlib.Path(__file__).parent / "schema.sql"
# Evidence + human-gate tables, kept separate so schema.sql stays a verbatim
# copy of the spec's Section 13.1. See schema_ext.sql for why.
_SCHEMA_EXT_PATH = pathlib.Path(__file__).parent / "schema_ext.sql"


class Store:
    def __init__(self, path: str = "saresq.db"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA_PATH.read_text())
        self.conn.executescript(_SCHEMA_EXT_PATH.read_text())
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

    # ------------------------------------------------------------------
    # media (evidence artefacts)
    # ------------------------------------------------------------------
    # Cheapest-and-most-useful-first. A 3 KB thumbnail of a 0.96-confidence
    # track is worth more to an operator than a 150 KB clip of a 0.3, so kind
    # dominates the ordering and confidence breaks ties within a kind.
    _KIND_RANK = (
        "CASE kind WHEN 'thumb' THEN 0 WHEN 'rgb_crop' THEN 1 "
        "WHEN 'thermal_patch' THEN 2 WHEN 'clip' THEN 3 ELSE 4 END"
    )

    def insert_media(self, **fields) -> int:
        return self._insert("media", **fields)

    def get_media(self, media_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM media WHERE media_id = ?", (media_id,)).fetchone()
        return dict(row) if row else None

    def media_for_target(self, target_id: int) -> list[dict]:
        rows = self.conn.execute(
            f"SELECT * FROM media WHERE target_id = ? ORDER BY {self._KIND_RANK}, media_id",
            (target_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def pending_media(self, limit: int | None = None) -> list[dict]:
        """Unsynced media, in the order the sync agent should send them."""
        sql = (
            f"SELECT * FROM media WHERE synced_ns IS NULL "
            f"ORDER BY {self._KIND_RANK} ASC, priority DESC, media_id ASC"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.conn.execute(sql).fetchall()]

    def set_media_progress(self, media_id: int, sent_bytes: int) -> None:
        self.conn.execute("UPDATE media SET sent_bytes = ? WHERE media_id = ?", (sent_bytes, media_id))
        self.conn.commit()

    def mark_media_synced(self, media_id: int, t_ns: int) -> None:
        self.conn.execute(
            "UPDATE media SET synced_ns = ?, sent_bytes = bytes WHERE media_id = ?", (t_ns, media_id)
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # alerts (Tier-1 telemetry-radio traffic)
    # ------------------------------------------------------------------
    def insert_alert(self, target_id: int, t_ns: int, payload: bytes) -> int:
        return self._insert("alerts", target_id=target_id, t_ns=t_ns, payload=payload)

    def pending_alerts(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM alerts WHERE sent_ns IS NULL ORDER BY alert_id"
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_alert_sent(self, alert_id: int, t_ns: int) -> None:
        self.conn.execute("UPDATE alerts SET sent_ns = ? WHERE alert_id = ?", (t_ns, alert_id))
        self.conn.commit()

    # ------------------------------------------------------------------
    # fusion features (frozen at capture; the labels' other half)
    # ------------------------------------------------------------------
    def insert_pass_features(self, pass_id: int, target_id: int, t_ns: int, features) -> int:
        import json

        return self._insert(
            "pass_features", pass_id=pass_id, target_id=target_id, t_ns=t_ns,
            features_json=json.dumps([float(x) for x in features]),
        )

    def latest_features_for_target(self, target_id: int) -> list[float] | None:
        """The most recent pass's feature vector -- what the fusion head last
        saw for this track, and therefore what an operator's verdict labels."""
        import json

        row = self.conn.execute(
            "SELECT features_json FROM pass_features WHERE target_id = ? "
            "ORDER BY t_ns DESC, pass_id DESC LIMIT 1",
            (target_id,),
        ).fetchone()
        return json.loads(row["features_json"]) if row else None

    # ------------------------------------------------------------------
    # verdicts (the human gate)
    # ------------------------------------------------------------------
    def insert_verdict(self, **fields) -> int:
        """Record an operator judgement. Deliberately does NOT touch the
        target's p_final or decision -- the machine's belief is preserved so
        the two can be compared afterwards."""
        return self._insert("verdicts", **fields)

    def verdicts_for_target(self, target_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM verdicts WHERE target_id = ? ORDER BY t_ns", (target_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def latest_verdict(self, target_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM verdicts WHERE target_id = ? ORDER BY t_ns DESC, verdict_id DESC LIMIT 1",
            (target_id,),
        ).fetchone()
        return dict(row) if row else None

    def all_verdicts(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM verdicts ORDER BY t_ns").fetchall()
        return [dict(r) for r in rows]

    def review_queue(self) -> list[dict]:
        """Targets with no verdict yet, most-confident first -- an operator's
        attention is the scarcest resource in the loop, so spend it top-down."""
        rows = self.conn.execute(
            "SELECT t.* FROM targets t "
            "LEFT JOIN verdicts v ON v.target_id = t.target_id "
            "WHERE v.verdict_id IS NULL "
            "ORDER BY t.p_final DESC, t.target_id"
        ).fetchall()
        return [dict(r) for r in rows]
