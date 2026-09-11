#!/usr/bin/env bash
# Keep the ML pipeline alive across an unattended stretch.
#
#   nohup bash training/watchdog.sh > results/watchdog.log 2>&1 &
#
# run_all.sh uses `set -e`, so ANY failing step aborts the remaining ones. Over
# a six-hour window with nobody watching, a single transient failure (an MPS
# hiccup, a memory spike, a step whose inputs were not ready) would otherwise
# waste the whole window. run_all.sh is idempotent and skips completed steps, so
# restarting it is always safe and always resumes.
#
# It gives up after MAX_RESTARTS so a genuinely broken step fails loudly instead
# of looping forever and burning the battery on the same traceback.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=results/run_all2.log
MAX_RESTARTS="${MAX_RESTARTS:-6}"
restarts=0

export EPOCHS="${EPOCHS:-30}" FT_EPOCHS="${FT_EPOCHS:-25}" BATCH="${BATCH:-16}" FT_FRACTION="${FT_FRACTION:-1.0}" \
       P2_BATCH="${P2_BATCH:-6}" WORKERS="${WORKERS:-4}" PATIENCE="${PATIENCE:-10}"

while true; do
  # run_all.sh prints this banner only after its last step.
  if grep -qa "===== ARTEFACTS =====" "$LOG" 2>/dev/null; then
    echo "[watchdog $(date '+%H:%M:%S')] pipeline COMPLETE"
    break
  fi

  if pgrep -f "bash training/run_all.sh" >/dev/null; then
    sleep 60
    continue
  fi

  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    echo "[watchdog $(date '+%H:%M:%S')] giving up after $restarts restarts."
    echo "[watchdog] last 30 lines of $LOG:"
    tail -30 "$LOG"
    break
  fi

  restarts=$((restarts + 1))
  echo "[watchdog $(date '+%H:%M:%S')] pipeline not running -- restart $restarts/$MAX_RESTARTS"
  bash training/run_all.sh >> "$LOG" 2>&1
  # Brief pause so an instantly-failing step cannot spin the loop.
  sleep 20
done
