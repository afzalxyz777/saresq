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

    def purge_mission(self) -> dict[str, int]:
        """Empty every per-mission table, keeping the schema.

        A mission is one flight of one payload. When the aircraft is power
        cycled it comes back with a new session token and everything before it
        belongs to a different sortie -- carrying those targets forward would
        put last flight's finds on this flight's map, which is the kind of
        mistake that sends a team to an empty building.

        Deletion order follows the foreign keys: media and pass_features point
        at passes, passes and alerts and verdicts point at targets.
        """
        counts: dict[str, int] = {}
        # rescores point at media, so they go first or the delete trips the
        # foreign key and silently leaves the new mission unable to store
        # evidence -- the exact failure mode that cost an evening already.
        for table in ("rescores", "media", "pass_features", "alerts", "verdicts",
                      "passes", "frames", "hazards", "targets"):
            try:
                cur = self.conn.execute(f"DELETE FROM {table}")
                counts[table] = cur.rowcount if cur.rowcount > 0 else 0
            except sqlite3.OperationalError:
                continue                      # table absent in this schema
        self.conn.commit()
        return counts

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
    # rescores (the ground station's second opinion)
    # ------------------------------------------------------------------
    def unscored_crops(self, limit: int = 16) -> list[dict]:
        """RGB crops that no rescore row references yet, newest first.

        Newest first on purpose: during a live mission the operator is looking
        at what just came in, so that is what should gain a second opinion
        first. A backlog from earlier in the flight is still worth scoring, but
        it is never what someone is waiting on.
        """
        rows = self.conn.execute(
            "SELECT m.* FROM media m "
            "LEFT JOIN rescores r ON r.media_id = m.media_id "
            "WHERE m.kind = 'rgb_crop' AND r.media_id IS NULL "
            "ORDER BY m.media_id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    def insert_rescore(self, **fields) -> int:
        """Idempotent by construction: UNIQUE(media_id) makes a second attempt
        a no-op rather than a duplicate, so a restart mid-batch is harmless."""
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        cur = self.conn.execute(
            f"INSERT OR IGNORE INTO rescores ({cols}) VALUES ({marks})",
            tuple(fields.values()),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def rescores_for_target(self, target_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM rescores WHERE target_id = ? ORDER BY p DESC", (target_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def best_rescore(self, target_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM rescores WHERE target_id = ? ORDER BY p DESC LIMIT 1",
            (target_id,),
        ).fetchone()
        return dict(row) if row else None

    def rescore_stats(self) -> dict:
        """Counts and the mean payload-vs-ground delta, for the status badge."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS n, AVG(ms) AS ms, AVG(p - p_payload) AS d_mean, "
            "SUM(CASE WHEN p > p_payload THEN 1 ELSE 0 END) AS n_up "
            "FROM rescores WHERE p_payload IS NOT NULL"
        ).fetchone()
        pend = self.conn.execute(
            "SELECT COUNT(*) AS n FROM media m "
            "LEFT JOIN rescores r ON r.media_id = m.media_id "
            "WHERE m.kind = 'rgb_crop' AND r.media_id IS NULL"
        ).fetchone()
        return {"scored": int(row["n"] or 0), "pending": int(pend["n"] or 0),
                "mean_ms": row["ms"], "delta_mean": row["d_mean"],
                "n_improved": int(row["n_up"] or 0)}

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
        attention is the scarcest resource in the loop, so spend it top-down.

        Ordering is the GATE, and it combines both opinions: the aircraft's
        p_final and the ground station's re-score of the same crop by a model
        it had no room to run. Two rules govern how they combine, and both are
        deliberate.

        ORDER BY THE HIGHER OF THE TWO. A second opinion may promote a
        candidate but may never bury one. The ground model is more precise on
        our own measurements (false-positive rate 0.013 against 0.020) and far
        more sensitive (0.560 against 0.020), so it is usually right -- but
        "usually right" is not the standard when being wrong means walking past
        a casualty. Promotion is cheap and reversible; demotion is neither.

        NOTHING IS EVER FILTERED OUT. This gate reorders attention, it does not
        remove candidates. An operator can always reach every target.

        `agreement` names the pattern, because the disagreements are the
        interesting rows: GROUND_HIGHER is a find the aircraft nearly let go,
        GROUND_LOWER is probably a false alarm and can wait.
        """
        rows = self.conn.execute(
            "SELECT t.*, r.p AS rescore_p, r.model AS rescore_model, "
            "       MAX(COALESCE(t.p_final, 0), COALESCE(r.p, 0)) AS gate_p "
            "FROM targets t "
            "LEFT JOIN verdicts v ON v.target_id = t.target_id "
            "LEFT JOIN (SELECT target_id, MAX(p) AS p, model FROM rescores "
            "           GROUP BY target_id) r ON r.target_id = t.target_id "
            "WHERE v.verdict_id IS NULL "
            "ORDER BY gate_p DESC, t.target_id"
        ).fetchall()

        out = []
        for row in rows:
            d = dict(row)
            pa, pg = d.get("p_final"), d.get("rescore_p")
            if pg is None:
                d["agreement"] = "NO_RESCORE"
            elif pa is None:
                d["agreement"] = "GROUND_ONLY"
            elif pg >= 0.35 and pa < 0.35:
                d["agreement"] = "GROUND_HIGHER"      # the aircraft nearly missed it
            elif pa >= 0.35 and pg < 0.15:
                d["agreement"] = "GROUND_LOWER"       # probably a false alarm
            else:
                d["agreement"] = "AGREE"
            out.append(d)
        return out
