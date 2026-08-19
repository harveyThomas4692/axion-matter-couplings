# moduli_scan/model2 — Kähler-moduli (t3) scan

Run the three-stage axion-matter-coupling pipeline for the second model across a grid of
the D-flat Kähler shape modulus `t3` (five network-initialisation seeds per point), and
reproduce the coupling figures. The stage scripts use the same calculation as the
`model2_*.ipynb` notebooks at the repository root; this directory adds the scan harness,
the figure scripts, and the aggregated results so the figures can be regenerated without
re-running the scan.

Run all commands from the repository root.

## The scan variable

The SUSY (D-flat) locus is the two-parameter cone
`t2 = 1, t1 = (t3-2)(t3+1), t4 = t1/t3, t3 > 2` (one shape modulus `t3` after fixing the
overall scale); `test_dflat_locus.py` re-derives and verifies it independently, along
with the no-scale identity `t.g.t = 3/4` and the split into 2 eaten + 2 light axions.
The complex structure is fixed at `psi0=2, psi1=1`.

## Pipeline

For each `(t3, seed)` point:

1. **Point cloud** — `gen_cloud.py` builds the point cloud for a given `t3`. Unlike
   model 1 the cloud is per-`t3` (the Kähler moduli enter `basis.pickle`), shared across
   seeds. `check_clouds.py` validates every cloud against the exact CY volume.
2. **Three stages** — `stage1_train.py` → `stage2_matter.py` → `stage3_axion.py` train
   the metric / bundle / harmonic-form networks and compute the matter fields and the
   axion-matter couplings, writing one `results.json` per run.
   `run_point.sh <t3> <seed> [n_points] [epochs] [parity]` drives all three stages.
3. **Gate** — `check_convergence.py` parses the run logs and writes
   `data/convergence.json`; `runsel.py` is the shared run-selection gate every consumer
   goes through (convergence, epoch/point floors, parity arm).
4. **Aggregate** -- `aggregate.py` (diagonal couplings) and `aggregate_fcnc.py`
   (off-diagonal / flavour-changing entries) collect the per-run `results.json` into
   `data/*.json`.

`scan.sbatch` is an example SLURM array submission — adjust account, partition and paths
for your cluster, and set `PYTHON` to your `cymetric[jax]` interpreter.

## The built-in correctness benchmark

The dilation and no-scale identities force the **volume-axion** diagonal coupling to
`1/sqrt(6) = 0.408248` for every sector at every point of the locus, and its off-diagonal
entries to vanish identically. Both are computed per run, so the scan carries a per-point
gauge of its own numerical systematic, and the figure scripts draw it (the FCNC floor
figure plots the measured off-diagonal "analytic zero" as a shaded error floor under the
physical curves). Deviations from these values
are numerical, never physics — see `plots/scan_plot_style.py` for the derivation.

## Labels

Sector names follow the notebooks' MSSM convention (`labels.py`; runs stamped
`"label_convention": "mssm-2026-08-13"`). The 50 runs of the original grid pre-date the
stamp and are remapped on read — see `labels.py` for the exact (verified) map.

## Data

`data/` holds the aggregated summaries (`combined_parity=corrected.json`,
`combined_fcnc.json`, `convergence.json`) plus the per-run
`data/runs/<point>/results.json`. Generated scan outputs (point clouds, trained
networks, fresh run directories) land under `results/` and are not tracked.

## Figures

The scripts in `plots/` read `data/` and write PDFs to `figures/`:

```
python moduli_scan/model2/plots/plot_couplings.py   # diagonal couplings vs t3 (benchmark, shape, per sector/generation)
python moduli_scan/model2/plots/plot_fcnc.py        # off-diagonal (FCNC) couplings vs t3 + error-floor figure
```

Pass `--paper` for the paper variants (trusted window `2.5 <= t3 <= 6`, print-size
fonts), `--no-latex` if no TeX installation is available.
