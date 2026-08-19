#!/usr/bin/env python3
"""Model 2: the MSSM field-label convention, and the remap between the two that exist.

WHY THIS MODULE EXISTS
----------------------
On 2026-08-14 the scan adopted the notebooks' MSSM naming for the model-2 matter
sectors. Before that it used a legacy naming, arising from how the Z2 parity of the
reference forms was repaired: the notebooks put a +1 in the parity exponent and kept
each form's field name, the scan had swapped the form specs between Q<->u^c/e^c and
d^c<->L and left the eps column alone.

The two conventions describe the SAME eight reference forms, hence the same ten trained
networks and the same couplings. They disagree only about which MSSM name is attached to
which form. That equivalence was machine-checked: the map below is derived from the two FIELD_MAP
tables, not assumed.

    legacy (pre-2026-08-14)        MSSM (adopted)
    -----------------------        --------------
    Q                      -->     u^c  and  e^c        (both: the tables set e^c = u^c)
    U                      -->     Q
    E                      -->     Q
    D                      -->     L
    L                      -->     D

Runs produced from 2026-08-14 stamp "label_convention": "mssm-2026-08-13" into their
results.json (stage3's [SCAN] serialization footer). The 50 runs
of the original grid pre-date that and carry no such key, so anything reading them must
remap. Every consumer of the run files goes through this module so
they cannot drift apart -- they read the same run files and must agree.

The remap RENAMES networks; it never recomputes or averages them. It is exact.
"""

# Written by stage3 from 2026-08-14 onward.
LABEL_CONVENTION = "mssm-2026-08-13"

# What a results.json without the key is assumed to be.
LEGACY_CONVENTION = "legacy-pre-2026-08-14"

# legacy sector name -> the MSSM sector name(s) holding the SAME forms.
LABEL_MAP_LEGACY_TO_MSSM = {"Q": ["U", "E"], "U": ["Q"], "E": ["Q"], "D": ["L"], "L": ["D"]}


def convention_of(results_dict):
    """Which label convention a loaded results.json is in."""
    return results_dict.get("label_convention", LEGACY_CONVENTION)


def relabel_sectors(sectors, where="<results.json>", allclose=None):
    """Rename pre-2026-08-14 sector keys into the MSSM convention.

    'U' and 'E' both map to his 'Q' and must agree numerically -- they are the same three
    forms under two names. That is CHECKED, not assumed: if they ever differ, something
    upstream of the naming is wrong and the run must not be aggregated.

    `allclose` is injected so this module needs no numpy import of its own; callers pass
    numpy.allclose. If None, the cross-check is done with exact equality.
    """
    out = {}
    for sec, sd in sectors.items():
        if sec not in LABEL_MAP_LEGACY_TO_MSSM:
            raise SystemExit(f"{where}: unknown sector name {sec!r}; cannot relabel safely")
        for new in LABEL_MAP_LEGACY_TO_MSSM[sec]:
            if new in out:
                # Second contributor to the same name: must be identical, not merged.
                a, b = out[new]["coupling_diag"], sd["coupling_diag"]
                for axion in a:
                    same = (a[axion] == b[axion] if allclose is None
                            else allclose(a[axion], b[axion], rtol=1e-12, atol=1e-14))
                    if not same:
                        raise SystemExit(
                            f"{where}: sectors mapping to '{new}' disagree on axion "
                            f"'{axion}' ({a[axion]} vs {b[axion]}). Both should be the "
                            f"same forms; refusing to guess which is right.")
                continue
            # Field names are the sector name plus a generation index, so they follow the
            # sector rename ('Q1' -> 'U1'; the universal 'D' -> 'L').
            renamed = dict(sd)
            renamed["fields"] = [new + f[len(sec):] for f in sd["fields"]]
            out[new] = renamed
    return out


def sectors_in_mssm_labels(results_dict, where="<results.json>", allclose=None):
    """Return this run's `sectors` dict with keys in the adopted convention."""
    if convention_of(results_dict) == LABEL_CONVENTION:
        return results_dict["sectors"]
    return relabel_sectors(results_dict["sectors"], where=where, allclose=allclose)
