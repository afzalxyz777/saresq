-- SaResQ evidence + human-gate schema.
--
-- This file is deliberately SEPARATE from schema.sql. schema.sql is the
-- Master Spec's Section 13.1 reproduced verbatim; everything here is beyond
-- the spec, added to support operator review and delay-tolerant sync. Keeping
-- them apart means you can always diff our additions against the spec.

-- ---------------------------------------------------------------------------
-- media: one row per stored evidence artefact.
--
-- Files are content-addressed on disk (blobs/<sha[:2]>/<sha>.<ext>) so an
-- identical crop is never written twice, but rows are NOT deduplicated -- two
-- targets may legitimately reference the same blob.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS media (
  media_id   INTEGER PRIMARY KEY,
  target_id  INTEGER REFERENCES targets(target_id),
  pass_id    INTEGER REFERENCES passes(pass_id),
  t_ns       INTEGER NOT NULL,
  kind       TEXT    NOT NULL,   -- thumb | rgb_crop | thermal_patch | clip
  sha256     TEXT    NOT NULL,
  rel_path   TEXT    NOT NULL,   -- relative to the media root, so the DB stays portable
  bytes      INTEGER NOT NULL,
  width      INTEGER,
  height     INTEGER,
  priority   REAL    NOT NULL DEFAULT 0,  -- p_final at capture; drives sync order
  sent_bytes INTEGER NOT NULL DEFAULT 0,  -- resume offset; survives a link drop
  synced_ns  INTEGER                      -- NULL until the ground station acks
);
CREATE INDEX IF NOT EXISTS idx_media_target ON media(target_id);
CREATE INDEX IF NOT EXISTS idx_media_pending ON media(synced_ns, priority DESC);

-- ---------------------------------------------------------------------------
-- alerts: Tier-1 traffic. A few dozen bytes per target, sent over the
-- telemetry radio the instant a track crosses HIGH/MEDIUM, so the operator
-- gets a map pin long before any imagery arrives.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
  alert_id  INTEGER PRIMARY KEY,
  target_id INTEGER NOT NULL REFERENCES targets(target_id),
  t_ns      INTEGER NOT NULL,
  payload   BLOB    NOT NULL,
  sent_ns   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_alerts_pending ON alerts(sent_ns);

-- ---------------------------------------------------------------------------
-- pass_features: the 18-element fusion vector as it was at each pass, in
-- saresq.fuse.features.FEATURE_NAMES order.
--
-- Stored at capture time because evidence cannot be reconstructed afterwards:
-- the background statistics, the tracker's hit ratio and the ambient context
-- are all gone by the time an operator sits down to review. Without this the
-- human gate produces opinions; with it, it produces labelled training data.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pass_features (
  pass_id       INTEGER PRIMARY KEY REFERENCES passes(pass_id),
  target_id     INTEGER REFERENCES targets(target_id),
  t_ns          INTEGER,
  features_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pass_features_target ON pass_features(target_id);

-- ---------------------------------------------------------------------------
-- verdicts: the human gate.
--
-- A verdict NEVER overwrites targets.p_final or targets.decision. The machine's
-- belief is preserved exactly as it was so the two can always be compared --
-- that comparison is both the safety audit and the training signal.
--
-- features_json freezes the 18-element fusion vector (saresq.fuse.features
-- FEATURE_NAMES order) at the moment of judgement, so training/export_verdicts.py
-- can build a labelled set without having to reconstruct evidence after the fact.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS verdicts (
  verdict_id          INTEGER PRIMARY KEY,
  target_id           INTEGER NOT NULL REFERENCES targets(target_id),
  verdict             TEXT    NOT NULL,  -- SURVIVOR | NOT_SURVIVOR | UNSURE | DISPATCHED
  operator            TEXT    NOT NULL,
  t_ns                INTEGER NOT NULL,
  p_final_at_verdict  REAL,              -- what the machine believed when judged
  decision_at_verdict TEXT,
  note                TEXT,
  features_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_verdicts_target ON verdicts(target_id);
