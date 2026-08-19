#!/bin/bash
# Model 2: run one (t3, seed) scan point end to end.
# Usage: moduli_scan/model2/run_point.sh <t3> <seed> [n_points] [epochs] [parity]
set -euo pipefail
T3="$1"; SEED="$2"; NPTS="${3:-500000}"; EPOCHS="${4:-25}"; PARITY="${5:-corrected}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python}"
# Overridable so a smoke/calibration run can use its OWN tiny cloud and its own output dir
# instead of the production paths. Two hazards this avoids, both previously realised:
#   * stage 1 GENERATES a cloud when basis.pickle is absent, so pointing a small-n run at a
#     committed cloud dir that was somehow incomplete would overwrite a scan input;
#   * a preliminary run writing into the production runs/ dir lands exactly where
#     aggregate.py looks. Send smokes to results/moduli_scan/model2/prelim/.
CLOUD="${SCAN_CLOUD_DIR:-$REPO/results/moduli_scan/model2/clouds/t3=${T3}}"
RUN="${SCAN_RUN_DIR:-$REPO/moduli_scan/model2/data/runs/t3=${T3}_seed=${SEED}_parity=${PARITY}}"
mkdir -p "$RUN"

# --t3= equals-form for symmetry with model 1's --psi1= gotcha; t3 is always positive
# here so it is not strictly required, but keep the habit.
"$PY" "$REPO/moduli_scan/model2/stage1_train.py"  --t3="$T3" --seed "$SEED" \
      --n-points "$NPTS" --epochs "$EPOCHS" --parity "$PARITY" \
      --cloud-dir "$CLOUD" --out-dir "$RUN"
"$PY" "$REPO/moduli_scan/model2/stage2_matter.py" --run-dir "$RUN" --t3="$T3" --parity "$PARITY"
"$PY" "$REPO/moduli_scan/model2/stage3_axion.py"  --run-dir "$RUN" --t3="$T3" --seed "$SEED" \
      --n-points "$NPTS" --epochs "$EPOCHS" --parity "$PARITY"
