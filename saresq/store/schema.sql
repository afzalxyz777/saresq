-- SaResQ persistence schema (Master Spec v3.0, Section 13.1), verbatim.

CREATE TABLE IF NOT EXISTS targets (
  target_id INTEGER PRIMARY KEY,
  first_seen_ns INTEGER NOT NULL,
  last_seen_ns INTEGER NOT NULL,
  lat REAL, lon REAL, pos_err_m REAL,
  p_final REAL, class TEXT, -- HIGH / MEDIUM / LOW
  n_passes INTEGER, decision TEXT, -- CONFIRM / REJECT / LOG_AND_RESUME / REOBSERVE_LOWER
  thumb_path TEXT
);

CREATE TABLE IF NOT EXISTS passes (
  pass_id INTEGER PRIMARY KEY,
  target_id INTEGER REFERENCES targets(target_id),
  t_start_ns INTEGER, t_end_ns INTEGER,
  alt_band INTEGER, alt_m REAL, speed_mps REAL,
  p_pass REAL, lr REAL, weight REAL,
  p_rgb_max REAL, z_peak_max REAL, iou_max REAL, hits INTEGER,
  t_bg REAL, t_ambient REAL, lum REAL,
  p_flood REAL, p_fire REAL, p_collapse REAL
);

CREATE TABLE IF NOT EXISTS hazards (
  id INTEGER PRIMARY KEY, t_ns INTEGER, lat REAL, lon REAL,
  class TEXT, p REAL
);

CREATE TABLE IF NOT EXISTS frames (
  t_ns INTEGER PRIMARY KEY, lat REAL, lon REAL, alt_m REAL,
  roll REAL, pitch REAL, yaw REAL, t_bg REAL, t_ambient REAL, sigma REAL,
  z_max REAL, n_crops INTEGER, fallback INTEGER,
  ms_gate REAL, ms_detect REAL, ms_fuse REAL, cpu_temp REAL
);
