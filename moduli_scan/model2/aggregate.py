#!/usr/bin/env python3
"""Model 2: aggregate per-(t3, seed) results.json into combined.json.

Mirrors moduli_scan/model1/aggregate.py, but the scan variable is the D-flat Kahler shape modulus t3
rather than the complex-structure modulus psi1, and runs are additionally keyed by the
`parity` setting (the corrected vs retired Z2 parity tables) so that the corrected and
legacy arms can never be silently averaged together.

LABEL CONVENTIONS -- read this before trusting a sector name.
On 2026-08-14 the scan adopted the notebooks' MSSM naming. Both conventions describe the SAME eight
reference forms and therefore the same ten trained networks; they disagree only on which
name is attached to which form:

    legacy (pre-2026-08-14)    MSSM (adopted)
    -----------------------    --------------
    Q                    <->   u^c  and  e^c
    U, E                 <->   Q
    D                    <->   L
    L                    <->   D

Runs produced from 2026-08-14 write "label_convention": "mssm-2026-08-13" into
results.json. The 50 runs of the original grid pre-date that and carry no such key, so this
aggregator remaps them on read. Everything it emits is therefore in the MSSM convention.
The remap is exact, not an approximation: it renames networks, it does not recompute them.

Usage:
  python moduli_scan/model2/aggregate.py [--runs-dir moduli_scan/model2/data/runs]
                            [--out moduli_scan/model2/data/combined.json]
                            [--parity corrected] [--labels mssm|as-stored]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", dest="runs_dir", default="moduli_scan/model2/data/runs")
ap.add_argument("--out", default=None)
ap.add_argument("--parity", default="corrected", choices=["corrected", "legacy", "all"],
                help="which parity arm to aggregate; 'all' emits one file per arm. "
                     "('notebook' was renamed 'legacy' on 2026-08-14: after the upstream fix, "
                     "'the notebook' is the corrected one.)")
ap.add_argument("--labels", default="mssm", choices=["mssm", "as-stored"],
                help="sector naming of the OUTPUT. 'mssm' (default) remaps pre-2026-08-14 "
                     "runs onto the adopted convention; 'as-stored' disables the remap and "
                     "will mix conventions if the run set spans the change -- diagnostics only")
ap.add_argument("--min-epochs", type=int, default=25,
                help="skip runs trained for fewer epochs (guards against averaging smoke "
                     "or calibration runs into a production scan); -1 disables")
ap.add_argument("--min-points", type=int, default=500000,
                help="skip runs with a smaller point cloud; -1 disables")
ap.add_argument("--quality", default="moduli_scan/model2/data/convergence.json",
                help="convergence QC file from moduli_scan/model2/check_convergence.py; runs it rejects "
                     "(a net diverged, final loss > 1) are excluded. Pass '' to disable.")
args = ap.parse_args()

# The label remap lives in moduli_scan/model2/labels.py, and the run-selection gate in moduli_scan/model2/runsel.py,
# so that this script and the other consumers of the same run files (aggregate_fcnc.py)
# share one implementation and cannot drift apart.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import LABEL_CONVENTION, convention_of, sectors_in_mssm_labels  # noqa: E402
from runsel import accept, find_runs, load_rejected, parity_of  # noqa: E402

files = find_runs(args.runs_dir)

# Training-convergence gate. A diverged sigma net makes its whole run unusable, and
# averaging it into a t3 point corrupts the point rather than just widening its error bar.
REJECTED = load_rejected(args.quality)


def aggregate(paths, out_path):
    raw = {}          # (sector, axion) -> t3 -> list of per-field arrays
    forms = {}
    t3s, n_used, conventions, diags = set(), 0, set(), []
    labels_seen, n_relabelled = set(), [0]
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if not accept(p, d, REJECTED, args.min_epochs, args.min_points):
            continue
        t3 = float(d["t3"])
        t3s.add(t3)
        n_used += 1
        conventions.add(d.get("coupling_normalisation", "legacy(bare)"))
        diags.append((t3, d.get("diagnostics", {})))

        # Bring the sector names into one convention before anything is averaged.
        stored_labels = convention_of(d)
        labels_seen.add(stored_labels)
        sectors = d["sectors"]
        if args.labels == "mssm" and stored_labels != LABEL_CONVENTION:
            sectors = sectors_in_mssm_labels(d, where=p, allclose=np.allclose)
            n_relabelled[0] += 1

        for sec, sd in sectors.items():
            forms[sec] = sd["fields"]
            for axion, vals in sd["coupling_diag"].items():
                raw.setdefault((sec, axion), {}).setdefault(t3, []).append(
                    np.asarray(vals, dtype=float))

    grid = sorted(t3s)
    out = {"t3_grid": grid, "n_runs": n_used,
           "conventions": sorted(conventions),
           "label_convention": (LABEL_CONVENTION if args.labels == "mssm"
                                else "as-stored (MIXED if more than one below)"),
           "label_conventions_of_inputs": sorted(labels_seen),
           "n_runs_relabelled": n_relabelled[0],
           "sectors": {}}
    for (sec, axion), by_t3 in raw.items():
        entry = out["sectors"].setdefault(sec, {"fields": forms[sec], "by_axion": {}})
        xs = sorted(by_t3)
        means, stds, ns = [], [], []
        for x in xs:
            stack = np.stack(by_t3[x], axis=0)
            means.append(stack.mean(0).tolist())
            stds.append((stack.std(0, ddof=1) if stack.shape[0] > 1
                         else np.zeros(stack.shape[1])).tolist())
            ns.append(int(stack.shape[0]))
        entry["by_axion"][axion] = {"t3": xs, "mean": means, "std": stds, "n": ns}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"  wrote {out_path}: {n_used} runs, t3 grid {grid}")
    print(f"  normalisation conventions present: {sorted(conventions)}")
    print(f"  label conventions of inputs: {sorted(labels_seen)}")
    if args.labels == "mssm":
        print(f"  sector names emitted in the '{LABEL_CONVENTION}' convention; "
              f"{n_relabelled[0]} run(s) remapped on read "
              f"(our Q -> his u^c/e^c, our U/E -> his Q, our D <-> his L)")
    elif len(labels_seen) > 1:
        print(f"  !! --labels as-stored with {len(labels_seen)} conventions present: "
              f"THIS OUTPUT MIXES NAMING AND MUST NOT BE PLOTTED")
    # Surface the on-locus diagnostics; these must hold at EVERY scan point.
    bad = [(t, dg) for t, dg in diags
           if dg and (abs(dg.get("no_scale_t_g_t", 0.75) - 0.75) > 1e-9
                      or max(dg.get("vol_dot_eaten", [0]) + dg.get("shape_dot_eaten", [0])) > 1e-8)]
    if bad:
        print(f"  !! {len(bad)} run(s) FAIL the on-locus diagnostics:")
        for t, dg in bad[:5]:
            print(f"     t3={t}: {dg}")
    else:
        print(f"  on-locus diagnostics OK for all {len(diags)} runs "
              f"(t.g.t=3/4; volume and shape both g-orthogonal to the eaten span)")
    for sec in out["sectors"]:
        print(f"    sector {sec:3s} fields={out['sectors'][sec]['fields']} "
              f"axions={list(out['sectors'][sec]['by_axion'])}")


arms = ["corrected", "legacy"] if args.parity == "all" else [args.parity]
for arm in arms:
    sel = [p for p in files if parity_of(p) == arm]
    if not sel:
        print(f"(no runs for parity={arm})")
        continue
    default_out = f"moduli_scan/model2/data/combined_parity={arm}.json"
    print(f"parity={arm}:")
    aggregate(sel, args.out or default_out)
