#!/usr/bin/env python3
"""Model 2: the ONE definition of "which runs count", shared by every aggregator.

WHY THIS MODULE EXISTS
----------------------
On 2026-08-14 a gap was found and closed: a second consumer read the per-run
results.json directly rather than through aggregate.py, so a change applied in one place
silently did not reach the other and two committed outputs disagreed with nothing to show
it. The label remap was moved into moduli_scan/model2/labels.py for exactly that reason. The run
SELECTION gate had the same shape of exposure -- a convergence rejection, an epoch floor
and a point-count floor, each re-implemented per consumer -- so it lives here now.

The gate, in order:
  * parity arm       -- from the run-dir name (moduli_scan/model2/run_point.sh writes it). The
                        corrected and legacy arms must never be averaged together.
  * convergence      -- runs rejected by moduli_scan/model2/check_convergence.py (a diverged net,
                        final loss > 1). A diverged sigma net corrupts a t3 point rather
                        than merely widening its error bar, so it is excluded, not
                        downweighted.
  * epochs / points  -- floors that keep smoke and calibration runs out of a production
                        aggregate. This is structural: a calibration once wrote
                        results.json to exactly the path a production point occupies.

Nothing here knows about couplings; it decides only which files are in.
"""
import glob
import json
import os
import re

# moduli_scan/model2/run_point.sh encodes the parity arm in the run-directory name.
_PAR_RE = re.compile(r"_parity=([A-Za-z]+)$")

# Production settings of the model-2 scan -- the floors below which a
# run is smoke, not science.
DEFAULT_MIN_EPOCHS = 25
DEFAULT_MIN_POINTS = 500000
DEFAULT_QUALITY = "moduli_scan/model2/data/convergence.json"


def parity_of(path):
    """Parity arm of the run holding `path` (a results.json), or 'unknown'."""
    m = _PAR_RE.search(os.path.basename(os.path.dirname(path)))
    return m.group(1) if m else "unknown"


def run_key(d):
    """The (t3, seed) key used by moduli_scan/model2/check_convergence.py's reject list."""
    return f"t3={float(d['t3'])}_seed={int(d['seed'])}"


def load_rejected(quality=DEFAULT_QUALITY, verbose=True):
    """The set of run keys the convergence gate rejects. Empty set if disabled."""
    if not quality:
        return set()
    if not os.path.exists(quality):
        if verbose:
            print(f"quality gate: {quality} NOT FOUND -- no convergence filtering applied. "
                  f"Run moduli_scan/model2/check_convergence.py first.")
        return set()
    with open(quality) as f:
        q = json.load(f)
    rejected = set(q.get("rejected", []))
    if verbose:
        print(f"quality gate: {quality} rejects {len(rejected)} run(s): {sorted(rejected)}")
    return rejected


def find_runs(runs_dir="moduli_scan/model2/data/runs"):
    """Sorted results.json paths under `runs_dir`. Raises if there are none."""
    files = sorted(glob.glob(os.path.join(runs_dir, "*", "results.json")))
    if not files:
        raise SystemExit(f"no results.json under {runs_dir}")
    return files


def accept(path, d, rejected, min_epochs=DEFAULT_MIN_EPOCHS,
           min_points=DEFAULT_MIN_POINTS, verbose=True):
    """Should this loaded run be aggregated? Prints the reason when it is not.

    Kept as a predicate over an ALREADY-LOADED dict so callers that need the payload do
    not read the file twice, and so the reasons print in the same order and wording for
    every consumer.
    """
    if run_key(d) in rejected:
        if verbose:
            print(f"  SKIP {path}: rejected by the convergence gate ({run_key(d)})")
        return False
    ep = int(d.get("epochs", -1))
    npt = int(d.get("n_points", -1))
    if min_epochs > 0 and 0 <= ep < min_epochs:
        if verbose:
            print(f"  SKIP {path}: epochs={ep} < {min_epochs}")
        return False
    if min_points > 0 and 0 <= npt < min_points:
        if verbose:
            print(f"  SKIP {path}: n_points={npt} < {min_points}")
        return False
    if ep < 0 or npt < 0:
        if verbose:
            print(f"  WARNING {path}: no epochs/n_points provenance (pre-dates the guard)")
    return True
