#!/bin/bash
# Example SLURM submission; adjust account/partition/qos/paths for your cluster.
# The (t3 x seed) Kahler scan, one node per point. task_id = 5*t3_idx + seed.
# Run ONE point first (moduli_scan/model2/run_point.sh 3.0 0) to measure cost, then set --time.
#SBATCH --job-name=m2scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --requeue
#SBATCH --array=0-49
#SBATCH --output=logs/slurm/m2scan-%A_%a.out
#SBATCH --error=logs/slurm/m2scan-%A_%a.err
set -euo pipefail
# Portable: submit from the repo root, or set SCAN_REPO.
REPO="${SCAN_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2
T3_VALUES=(2.25 2.5 3.0 3.5 4.0 5.0 6.0 8.0 10.0 14.0)
NUM_SEEDS=5
t3_idx=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
seed=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
t3=${T3_VALUES[$t3_idx]}
echo "=== scan t3=$t3 seed=$seed host=$(hostname) $(date) ==="
bash moduli_scan/model2/run_point.sh "$t3" "$seed" 500000 25 corrected
echo "=== done t3=$t3 seed=$seed $(date) ==="
