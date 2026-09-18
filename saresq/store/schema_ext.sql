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

-- ---------------------------------------------------------------------------
-- rescores: the ground station's second opinion on an evidence crop.
--
-- The payload must fit a 3 MB nano detector into 1 GB and answers in 28 ms on
-- a 160 px crop. The ground station has no such budget, so once a crop has
-- been uploaded it is scored again by a far larger model at a far larger input
-- size. This is the same cascade principle the payload runs internally --
-- spend the expensive stage only on what the cheap one already nominated --
-- extended one hop across the radio link.
--
-- Like verdicts, a rescore NEVER overwrites the payload's own number. Both are
-- kept so the two can be compared: that comparison is the measurement of what
-- the second opinion is worth, and without it "bigger model, better answer" is
-- an assertion rather than a result.
--
-- UNIQUE(media_id) is what makes the worker idempotent. Re-running it, or
-- restarting the ground station mid-mission, can never double-score a crop or
-- leave a partial pass behind, because the database refuses the duplicate
-- rather than the worker having to remember what it has already done.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rescores (
  rescore_id INTEGER PRIMARY KEY,
  media_id   INTEGER NOT NULL UNIQUE REFERENCES media(media_id),
  target_id  INTEGER REFERENCES targets(target_id),
  t_ns       INTEGER NOT NULL,
  p          REAL    NOT NULL,     -- best person confidence in the crop
  n          INTEGER NOT NULL,     -- person boxes above threshold
  p_payload  REAL,                 -- what the aircraft said, frozen for comparison
  model      TEXT    NOT NULL,
  imgsz      INTEGER NOT NULL,
  ms         REAL
);
CREATE INDEX IF NOT EXISTS idx_rescores_target ON rescores(target_id);
