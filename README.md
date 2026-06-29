# axion-matter-couplings
The code for our paper on calculating Axion-Matter-Couplings from first principles. 

Find the preprint for our paper on ArXiv: https://arxiv.org/abs/xxxx.xxxxx

This makes use of the cymetric package: https://github.com/ruehlef/cymetric

## Repository layout

```
*.ipynb        notebooks: the metric/Yukawa, matter-field and axion-coupling calculations
moduli_scan/   a self-contained sub-project that runs the notebook calculation across a
               grid of complex-structure parameters psi1 (multiple seeds) and reproduces
               the coupling figures — harness, plot scripts and aggregated data together
```

The three notebooks at the repository root are the method: the metric/Yukawa
computation, the matter-field construction, and the axion-matter couplings. The
`moduli_scan/` directory builds on that same code to scan over the complex-structure
modulus psi1; see [`moduli_scan/README.md`](moduli_scan/README.md) for its quickstart.

```
moduli_scan/
  stage1_yukawa.py stage2_matter.py stage3_axion.py   three-stage per-point pipeline
  gen_cloud.py  run_point.sh  patch_cymetric.py        cloud generation + drivers
  aggregate*.py  fcnc_lib.py                            collect per-run results
  scan.sbatch  scan_complex.sbatch                      example SLURM submissions
  plots/                                                figure scripts (+ shared style)
  data/                                                 combined*.json + per-run results.json
```

## Reproducing the scan

Run the commands from the repository root. The scan runs the three-stage pipeline for
each (psi1, seed) point and writes a `results.json` per run:

1. `moduli_scan/gen_cloud.py` — generate the shared point cloud for a given psi1 (one per
   psi1, reused across seeds).
2. `moduli_scan/stage1_yukawa.py` → `stage2_matter.py` → `stage3_axion.py` — train the
   networks and compute the matter fields and axion-matter couplings.
   `moduli_scan/run_point.sh <psi1> <seed>` drives all three stages for one point.
3. `moduli_scan/aggregate.py`, `aggregate_complex.py`, `aggregate_fcnc.py` — collect the
   per-run `results.json` into the `moduli_scan/data/combined*.json` summaries.

Requirements: `cymetric` with the JAX backend (`pip install "cymetric[jax] @
git+https://github.com/ruehlef/cymetric.git"`), plus `jax`, `equinox`, `optax`. On some
platforms run `python moduli_scan/patch_cymetric.py` once after installing cymetric (a
compatibility shim for the JAX trainer; see the file header).

`moduli_scan/scan.sbatch` and `moduli_scan/scan_complex.sbatch` are example SLURM array
submissions for the real-axis and complex-psi1 grids — adjust the account, partition and
paths for your cluster. Set `PYTHON` to your cymetric[jax] interpreter if `python` is not
it.

## Figures

The scripts in `moduli_scan/plots/` read the aggregated JSON in `moduli_scan/data/` and
write PDFs to `figures/`:

```
python moduli_scan/plots/plot_couplings.py      # couplings vs psi1, per sector (real-axis scan)
python moduli_scan/plots/plot_fcnc.py           # off-diagonal (FCNC) couplings vs psi1
python moduli_scan/plots/plot_complex.py        # complex-psi1 maps + CP-proxy null check
python moduli_scan/plots/plot_complex_diag.py   # complex-psi1 diagonal-coupling plane maps
```

Each accepts `--no-latex` to disable LaTeX rendering if a TeX install is not available.
The shipped `moduli_scan/data/` lets the figures regenerate without re-running the scan.
