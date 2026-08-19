# moduli_scan/model1 — complex-structure (psi1) scan

Run the three-stage axion-matter-coupling pipeline across a grid of complex-structure
parameters `psi1` (with multiple network-initialisation seeds) and reproduce the coupling
figures. The stage scripts use the same calculation as the root notebooks; this directory
adds the scan harness, the figure scripts, and the aggregated results so the figures can
be regenerated without re-running the scan.

Run all commands from the repository root.

## Pipeline

For each `(psi1, seed)` point:

1. **Point cloud** — `gen_cloud.py` builds the shared point cloud for a given `psi1` (one
   cloud per `psi1`, reused across seeds).
2. **Three stages** — `stage1_yukawa.py` → `stage2_matter.py` → `stage3_axion.py` train the
   metric / bundle / harmonic-form networks and compute the matter fields and the
   axion-matter couplings, writing one `results.json` per run.
   `run_point.sh <psi1> <seed> [n_points] [epochs]` drives all three stages for one point.
3. **Aggregate** — `aggregate.py` (real-axis), `aggregate_complex.py` (complex `psi1`) and
   `aggregate_fcnc.py` (off-diagonal / flavour-changing entries) collect the per-run
   `results.json` into the `moduli_scan/model1/data/combined*.json` summaries.

`patch_cymetric.py` is a one-time compatibility shim for cymetric's JAX trainer (see its
header). `scan.sbatch` / `scan_complex.sbatch` are example SLURM array submissions —
adjust account, partition and paths for your cluster, and set `PYTHON` to your
`cymetric[jax]` interpreter.

## Data

`moduli_scan/model1/data/` holds the aggregated `combined*.json` summaries plus the per-run
`moduli_scan/model1/data/runs/<point>/results.json`. Generated scan outputs (point clouds, trained networks,
fresh run directories) land under `results/` and are not tracked.

## Figures

The scripts in `plots/` read `moduli_scan/model1/data/` and write PDFs to `figures/`:

```
python moduli_scan/model1/plots/plot_couplings.py      # couplings vs psi1, per sector (real axis)
python moduli_scan/model1/plots/plot_fcnc.py           # off-diagonal (FCNC) couplings vs psi1
python moduli_scan/model1/plots/plot_complex.py        # complex-psi1 maps + CP-proxy null check
python moduli_scan/model1/plots/plot_complex_diag.py   # complex-psi1 diagonal-coupling plane maps
```

The real-axis plots read only the `combined*.json` summaries; the complex-`psi1` plots
reconstruct the coupling matrices from the per-run `moduli_scan/model1/data/runs/` files. Pass `--no-latex`
to disable LaTeX rendering if no TeX install is available.
