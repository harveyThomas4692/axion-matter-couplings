#!/bin/bash
# Example SLURM submission; adjust account/partition/paths for your cluster.
#
# Complex-psi1 scan: 64 jobs = 20 first-quadrant grid points x 3 seeds (=60)
# + 4 symmetry-probe points x 1 seed (=4). The real axis Im(psi1)=0 is covered
# separately by the real scan (moduli_scan/scan.sbatch).
#
#   grid  = Re{0,1,2,4,6} x Im{0.5,1,2,4}                (20 values, seeds 0-2)
#   probe = {1-1j, -1+1j, -1-1j, 2-2j}                   (4 values, seed 0)
#           -> test CP (psi->psibar => M->M*) and psi->-psi reflection symmetries
#
# A flat JOBS list maps task_id (0..63) -> "psi1:seed" (no modular ambiguity).
# Each task: stage1 (train, COMPLEX psi1, reuse pre-gen cloud) -> stage2 -> stage3.
# psi1 only changes the geometry, not the per-point work.
#
# The seeds of each psi1 SHARE one cloud -> pre-generate each cloud once with
# moduli_scan/gen_cloud.py (per psi1) before submitting, so the seeds do not race.
#
# Output:  results/moduli_scan/runs/psi1=<v>_seed=<s>/results.json  (with psi1_re/psi1_im)

#SBATCH --job-name=axscanC
#SBATCH --partition=lr8           # EDIT: your CPU partition
#SBATCH --account=your_account    # EDIT: your allocation
#SBATCH --qos=lr8_normal          # EDIT: your QOS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --array=0-63
#SBATCH --output=logs/slurm/axscanC-%A_%a.out
#SBATCH --error=logs/slurm/axscanC-%A_%a.err

set -euo pipefail
REPO="${PROJECT_ROOT:-$(pwd)}"    # EDIT: repo root (default: submission dir)
cd "$REPO"

# MUST match gen_cloud_complex.sbatch.
PSI_GRID=(
  "0+0.5j" "0+1j" "0+2j" "0+4j"
  "1+0.5j" "1+1j" "1+2j" "1+4j"
  "2+0.5j" "2+1j" "2+2j" "2+4j"
  "4+0.5j" "4+1j" "4+2j" "4+4j"
  "6+0.5j" "6+1j" "6+2j" "6+4j"
)
PSI_PROBE=( "1-1j" "-1+1j" "-1-1j" "2-2j" )

# Flat (psi1:seed) job list: 20 grid x 3 seeds, then 4 probe x 1 seed = 64.
JOBS=()
for p in "${PSI_GRID[@]}"; do for s in 0 1 2; do JOBS+=("$p:$s"); done; done
for p in "${PSI_PROBE[@]}"; do JOBS+=("$p:0"); done

spec=${JOBS[$SLURM_ARRAY_TASK_ID]}
psi1=${spec%:*}
seed=${spec#*:}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2

echo "=== axscanC task=$SLURM_ARRAY_TASK_ID psi1=$psi1 seed=$seed host=$(hostname) $(date) ==="
bash moduli_scan/run_point.sh "$psi1" "$seed" 500000 25
echo "=== done task=$SLURM_ARRAY_TASK_ID $(date) ==="
