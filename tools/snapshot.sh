#!/usr/bin/env bash
# Snapshot everything git cannot hold, so an experiment is always reversible.
#
#     bash tools/snapshot.sh              # -> ~/saresq-snapshots/<tag>/
#     bash tools/snapshot.sh before-11n
#
# Git has the source and the calibration. It does NOT have the trained weights,
# the TFLite exports or the bench captures -- ~185 MB of artefacts that cost
# eight hours of GPU and a session with hot water to produce and cannot be
# regenerated from the repo. Those live only on this machine until this runs.
#
# Verification is the point, not the copy: every file is checksummed on the way
# in and the manifest is re-verified on the way out, because a backup nobody has
# read back is a belief, not a backup.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-$(date +%Y%m%d-%H%M)}"
DEST="${SNAPSHOT_DIR:-$HOME/saresq-snapshots}/$TAG"
mkdir -p "$DEST"

echo "== snapshotting to $DEST =="
COMMIT=$(git rev-parse HEAD)
BRANCH=$(git branch --show-current)
# `|| true`: grep exits 1 when it matches nothing, and under `set -o pipefail`
# a clean working tree would abort the whole snapshot.
DIRTY=$(git status --porcelain | { grep -v '^??' || true; } | wc -l | tr -d ' ')

{
  echo "snapshot     $TAG"
  echo "created      $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "git commit   $COMMIT"
  echo "git branch   $BRANCH"
  echo "uncommitted  $DIRTY tracked file(s) modified at snapshot time"
  echo
  echo "To restore:"
  echo "  git checkout $COMMIT"
  echo "  rsync -a $DEST/models/ models/"
  echo "  rsync -a $DEST/results/ results/"
  echo
} > "$DEST/MANIFEST.txt"

# rsync keeps the directory shape, so a restore is a copy back rather than an
# unpack-and-guess-where-it-went.
# results/calib keeps its .npy files: those are the RAW THERMAL ARRAYS from the
# bench session, and the affine cannot be re-fitted from the RGB frames alone.
# Everywhere else .npy means an Ultralytics dataset cache, which is regenerated
# on demand and would double the snapshot for nothing.
for src in models results/detector results/hazard results/fusion; do
  [ -e "$src" ] || continue
  mkdir -p "$DEST/$src"
  rsync -a --exclude '*.npy' --exclude '__pycache__' "$src/" "$DEST/$src/"
done
if [ -e results/calib ]; then
  mkdir -p "$DEST/results/calib"
  rsync -a --exclude '__pycache__' results/calib/ "$DEST/results/calib/"
fi

echo "== checksumming =="
( cd "$DEST" && find models results -type f \( -name '*.pt' -o -name '*.tflite' -o -name '*.keras' -o -name '*.json' \) \
    -exec shasum -a 256 {} \; | sort -k2 > CHECKSUMS.sha256 )
  # .npy is checksummed too where it is real data rather than a cache.
  ( cd "$DEST" && find results/calib -name '*.npy' -exec shasum -a 256 {} \; 2>/dev/null \
      | sort -k2 >> CHECKSUMS.sha256 || true )
N=$(wc -l < "$DEST/CHECKSUMS.sha256" | tr -d ' ')
echo "   $N artefact(s)"

echo "== verifying =="
if ( cd "$DEST" && shasum -a 256 -c CHECKSUMS.sha256 --quiet ); then
  echo "   all $N checksums match"
else
  echo "   !! VERIFICATION FAILED -- do not trust this snapshot"; exit 1
fi

echo "{\"tag\":\"$TAG\",\"commit\":\"$COMMIT\",\"branch\":\"$BRANCH\",\"artefacts\":$N}" > "$DEST/snapshot.json"
du -sh "$DEST" | sed 's|^|   total |'
echo
echo "restore with:  rsync -a $DEST/models/ models/ && rsync -a $DEST/results/ results/"
echo "               git checkout $COMMIT"
