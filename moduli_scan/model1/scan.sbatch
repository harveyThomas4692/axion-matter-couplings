#!/bin/bash
# Example SLURM submission; adjust account/partition/paths for your cluster.
#
# The (psi1 x seed) scan, one node per (psi1, seed).
#   task_id = 5 * psi_idx + seed,  psi_idx in {0..10}, seed in {0..4}.
# Each task: stage1 (train, reuse shared cloud) -> stage2 -> stage3 -> results.json.
# stage1 generates a missing per-psi1 cloud itself; pre-generating the clouds once
# (moduli_scan/model1/gen_cloud.py per psi1) avoids multiple seeds racing to build the same cloud.
#
# Output:  results/moduli_scan/model1/runs/psi1=<v>_seed=<s>/results.json
#
# Run ONE point first (moduli_scan/model1/run_point.sh 1.0 0) to measure cost, then set --time.

#SBATCH --job-name=axscan
#SBATCH --partition=lr8           # EDIT: your CPU partition
#SBATCH --account=your_account    # EDIT: your allocation
#SBATCH --qos=lr8_normal          # EDIT: your QOS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-54
#SBATCH --output=logs/slurm/axscan-%A_%a.out
#SBATCH --error=logs/slurm/axscan-%A_%a.err

set -euo pipefail
REPO="${PROJECT_ROOT:-$(pwd)}"    # EDIT: repo root (default: submission dir)
cd "$REPO"

PSI_VALUES=(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0)
NUM_SEEDS=5
psi_idx=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
seed=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
psi1=${PSI_VALUES[$psi_idx]}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2

echo "=== axscan task=$SLURM_ARRAY_TASK_ID psi1=$psi1 seed=$seed host=$(hostname) $(date) ==="
bash moduli_scan/model1/run_point.sh "$psi1" "$seed" 500000 25
echo "=== done task=$SLURM_ARRAY_TASK_ID $(date) ==="
