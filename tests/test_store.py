"""Section 13.1 SQLite schema: basic CRUD and WAL durability setup."""
import sqlite3

from saresq.store.db import Store


def test_insert_and_read_back_a_target(tmp_path):
    with Store(str(tmp_path / "test.db")) as store:
        target_id = store.insert_target(
            first_seen_ns=1000, last_seen_ns=2000, lat=22.5, lon=88.3,
            pos_err_m=2.7, p_final=0.96, class_="HIGH", n_passes=2, decision="CONFIRM",
        )
        row = store.get_target(target_id)
        assert row["lat"] == 22.5
        assert row["decision"] == "CONFIRM"


def test_update_target_and_insert_passes(tmp_path):
    with Store(str(tmp_path / "test.db")) as store:
        target_id = store.insert_target(first_seen_ns=0, last_seen_ns=0, n_passes=0)
        store.insert_pass(target_id=target_id, alt_band=1, p_pass=0.45, weight=1.0)
        store.insert_pass(target_id=target_id, alt_band=0, p_pass=0.88, weight=1.0)
        store.update_target(target_id, n_passes=2, p_final=0.96, decision="CONFIRM")

        target = store.get_target(target_id)
        assert target["n_passes"] == 2
        passes = store.get_passes_for_target(target_id)
        assert len(passes) == 2
        assert passes[0]["p_pass"] == 0.45


def test_insert_hazard_handles_the_class_keyword_collision(tmp_path):
    with Store(str(tmp_path / "test.db")) as store:
        store.insert_hazard(t_ns=0, lat=22.57, lon=88.36, class_="flood", p=0.8)
        hazards = store.all_hazards()
        assert len(hazards) == 1
        assert hazards[0]["class"] == "flood"


def test_wal_mode_is_enabled(tmp_path):
    db_path = tmp_path / "test.db"
    with Store(str(db_path)):
        pass
    conn = sqlite3.connect(str(db_path))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
