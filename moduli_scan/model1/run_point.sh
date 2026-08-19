#!/bin/bash
# Driver: run ONE (psi1, seed) point through stages 1->2->3.
#   stage 1 trains nets (uses the shared per-psi1 cloud)  -> yukawa_run_*.pkl
#   stage 2 builds all matter fields                       -> matter_fields_all.pkl
#   stage 3 computes G/Lambda/couplings                    -> results.json
#
# Usage: moduli_scan/model1/run_point.sh <psi1> <seed> [n_points] [epochs]
# Set PYTHON to your cymetric[jax] interpreter if `python` is not the right one.
set -euo pipefail

PSI1="$1"; SEED="$2"; NPTS="${3:-500000}"; EPOCHS="${4:-25}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-python}"

CLOUD="$REPO/results/moduli_scan/model1/clouds/psi1=${PSI1}"
RUN="$REPO/results/moduli_scan/model1/runs/psi1=${PSI1}_seed=${SEED}"
mkdir -p "$RUN"

echo "=== run_point psi1=$PSI1 seed=$SEED npts=$NPTS epochs=$EPOCHS ==="
echo "cloud=$CLOUD"
echo "run=$RUN"

# Stage 1 will generate the cloud if gen_cloud.sbatch hasn't already (idempotent guard).
# NOTE: --psi1=... (equals form) is REQUIRED: a negative-real psi1 like "-1+1j"
# would otherwise be parsed by argparse as a flag ("expected one argument").
"$PY" "$REPO/moduli_scan/model1/stage1_yukawa.py" --psi1="$PSI1" --seed "$SEED" \
      --n-points "$NPTS" --epochs "$EPOCHS" --cloud-dir "$CLOUD" --out-dir "$RUN"

"$PY" "$REPO/moduli_scan/model1/stage2_matter.py" --run-dir "$RUN"

"$PY" "$REPO/moduli_scan/model1/stage3_axion.py" --run-dir "$RUN" --psi1="$PSI1" --seed "$SEED"

echo "=== DONE psi1=$PSI1 seed=$SEED -> $RUN/results.json ==="
