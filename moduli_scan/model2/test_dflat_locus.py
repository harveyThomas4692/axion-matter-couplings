#!/usr/bin/env python
"""Model 2: independent verification of the D-flat Kahler locus and the axion split.

Model 2 is the tetraquadric (P1)^4[2,2,2,2] with a freely-acting Z2 and the line-bundle
sum of `LBSStandardModels` entry Num={7862,2}, from the Oxford line-bundle-model database
  https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/linebundlemodels/index.html
Model data (bundles, Z2, spectrum, reference forms, D-flat recipe) is specified in the
model-2 notebooks at the repository root,
  model2_{train,matter_fields_all,axion_coupling_all}.ipynb.

Unlike model 1 -- whose SUSY locus is the single symmetric point t=(1,1,1,1), see
moduli_scan/model1/stage3_axion.py:84 -- model 2's SUSY locus is a 2-parameter D-flat cone, which is
what makes the axion-matter couplings depend on the Kahler moduli. This test verifies,
INDEPENDENTLY of the notebooks' algebra, that:

  CHECK 1  the slopes mu(L_a) derived from scratch match the note's quoted formulas
  CHECK 2  solving mu=0 from scratch reproduces the note's parameterisation of the locus
  CHECK 3  rank{k_a^i} = 2, i.e. 2 eaten + 2 light axions
  CHECK 4  the recipe lands on the locus and inside the Kahler cone across the t3 range
  CHECK 5  the no-scale identity t^i t^j g_ij = 3/4 holds at every locus point
  CHECK 6  k_a^i g_ij t^j = 0 identically ON the locus -- so the volume direction is
           always g-orthogonal to the eaten span, i.e. always light. [proved below]

CHECK 6 is the structural reason model 2's notebook may take eaten=span{k1,k4} first and
define light as the g-orthogonal complement, whereas model 1 declared the volume direction
light first and projected the k_a against it (moduli_scan/model1/stage3_axion.py:106-115). Both orderings
agree precisely because of CHECK 6, and CHECK 6 holds *only* on the D-flat locus.

  Proof.  V = (1/6) d_ijk t^i t^j t^k, so V_i = (1/2) d_ijk t^j t^k and V_ij = d_ijk t^k.
  The axion kinetic metric is g_ij = (1/4)( -V_ij/V + V_i V_j / V^2 )
  (moduli_scan/model1/stage3_axion.py:93,102; = (1/4) d_i d_j (-ln V)).  Then
      k_a^i V_ij t^j = d_ijk k_a^i t^j t^k = mu(L_a),
      k_a^i V_i      = (1/2) d_ijk k_a^i t^j t^k = mu(L_a)/2,
      V_j t^j        = (1/2) d_jkl t^j t^k t^l   = 3V.
  Hence
      k_a^i g_ij t^j = (1/4)( -mu/V + (mu/2)(3V)/V^2 )
                     = (1/4)(mu/V)( -1 + 3/2 )
                     = mu(L_a) / (8V),
  which vanishes iff mu(L_a) = 0, i.e. exactly on the D-flat locus.  []
  (CHECK 6 verifies both the vanishing on-locus and this closed form off-locus.)

Run:  python moduli_scan/model2/test_dflat_locus.py
Exit code 0 on success, 1 on any failed check.
"""
import sys
import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# Model data
# ---------------------------------------------------------------------------
# Line bundle sum, rows = the four P1 degrees, cols = L1..L5.  LBSStandardModels Num={7862,2};
# used verbatim in
# model2_matter_fields_all.ipynb cell m3 (`kmat`).
KMAT = np.array([[-1, -1, -1, 1, 2],
                 [-1, -1, -1, 2, 1],
                 [1, 1, 1, -3, 0],
                 [1, 1, 1, -1, -2]], dtype=float)

# Triple intersection numbers of the tetraquadric: d_ijk = 2 for pairwise-distinct i,j,k,
# else 0.  Same manifold as model 1 (arXiv:2402.01615); identical construction appears in
# moduli_scan/model1/stage3_axion.py:95-99 and in model2_matter_fields_all.ipynb cell m3.
D3 = np.zeros((4, 4, 4))
for _i in range(4):
    for _j in range(4):
        for _k in range(4):
            if len({_i, _j, _k}) == 3:
                D3[_i, _j, _k] = 2.0

# The two independent anomalous-U(1) directions.  k1=k2=k3, so the
# charge matrix has rank 2 and the eaten span is {k1, k4}
# (model2_axion_coupling_all.ipynb cell c7).
K1 = KMAT[:, 0].copy()
K4 = KMAT[:, 3].copy()

# Kahler cone of the tetraquadric = the positive orthant (model2 note sec 4).
# Scan range for the shape modulus: t3 > 2 is required for t1 > 0 (see CHECK 2).
T3_GRID = [2.001, 2.5, 3.0, 4.0, 6.0, 10.0, 20.0]

TOL_SLOPE = 1e-9      # absolute; slopes are O(1)-O(1e4) across the grid, see CHECK 4
TOL_IDENT = 1e-12     # no-scale and orthogonality identities are exact up to round-off


def dflat_point(t3, t2=1.0):
    """The note's parameterisation of the D-flat locus (sec 4; notebook cell m3).

    Scale gauge t2 = 1; t3 is the one free shape modulus.
    """
    t1 = (t3 - 2.0) * (t3 + 1.0)      # = t3^2 - t3 - 2, from mu(L4)=0 at t2=1
    t4 = t1 / t3                      # from mu(L1)=0 :  t1 t2 = t3 t4
    return np.array([t1, t2, t3, t4], dtype=float)


def slopes(t):
    """mu(L_a) = d_ijk k_a^i t^j t^k for a = 1..5."""
    return np.array([np.einsum('ijk,i,j,k->', D3, KMAT[:, a], t, t) for a in range(5)])


def volume(t):
    """V = (1/6) d_ijk t^i t^j t^k.  Matches moduli_scan/model1/stage3_axion.py:94 for the tetraquadric."""
    return float((1.0 / 6.0) * np.einsum('ijk,i,j,k->', D3, t, t, t))


def axion_metric(t):
    """g_ij = (1/4) d_i d_j (-ln V).  Identical form to moduli_scan/model1/stage3_axion.py:101-102."""
    V = volume(t)
    Vi = np.einsum('ijk,j,k->i', D3, t, t) / 2.0
    Vij = np.einsum('ijk,k->ij', D3, t)
    return 0.25 * (-Vij / V + np.outer(Vi, Vi) / V**2)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check1_symbolic_slopes(report):
    """Derive the slopes symbolically and compare to the note's quoted expressions."""
    t1, t2, t3, t4 = sp.symbols('t1 t2 t3 t4', positive=True)
    ts = [t1, t2, t3, t4]
    mus = []
    for a in range(5):
        e = sum(int(D3[i, j, k]) * int(KMAT[i, a]) * ts[j] * ts[k]
                for i in range(4) for j in range(4) for k in range(4))
        mus.append(sp.expand(e))

    # Quoted in model2_axion_matter_group_theory_and_computation.md sec 4:
    md_mu1 = 8 * (t1 * t2 - t3 * t4)
    md_mu4 = -4 * (4 * t1 * t2 - t1 * t3 + t1 * t4 + 2 * t2 * t4 - 3 * t3 * t4)

    ok1 = sp.simplify(mus[0] - md_mu1) == 0
    ok4 = sp.simplify(mus[3] - md_mu4) == 0
    ok5 = sp.simplify(mus[4] - (-3 * mus[0] - mus[3])) == 0
    ok_deg = all(sp.simplify(mus[a] - mus[0]) == 0 for a in (1, 2))   # k1=k2=k3

    report(f"  derived mu(L1) = {sp.factor(mus[0])}")
    report(f"  derived mu(L4) = {sp.factor(mus[3])}")
    report(f"  mu(L1) matches note's quoted form : {ok1}")
    report(f"  mu(L4) matches note's quoted form : {ok4}")
    report(f"  mu(L5) = -3 mu(L1) - mu(L4)       : {ok5}")
    report(f"  mu(L1) = mu(L2) = mu(L3)          : {ok_deg}")
    return ok1 and ok4 and ok5 and ok_deg, mus, ts


def check2_symbolic_locus(report, mus, ts):
    """Solve mu(L1)=mu(L4)=0 at t2=1 from scratch; compare to the note's recipe."""
    t1, t2, t3, t4 = ts
    sol = sp.solve([mus[0].subs(t2, 1), mus[3].subs(t2, 1)], [t1, t4], dict=True)
    if len(sol) != 1:
        report(f"  expected a unique solution branch, got {len(sol)}")
        return False
    s = sol[0]
    ok_t1 = sp.simplify(s[t1] - (t3 - 2) * (t3 + 1)) == 0
    ok_t4 = sp.simplify(s[t4] - s[t1] / t3) == 0
    report(f"  solved t1 = {sp.factor(sp.simplify(s[t1]))}   (recipe: (t3-2)(t3+1))  -> {ok_t1}")
    report(f"  solved t4 = {sp.simplify(s[t4])}   (recipe: t1/t3)          -> {ok_t4}")
    # t1 > 0 requires t3 > 2 (the other root t3 = -1 is outside the positive orthant)
    report("  => t1 > 0 iff t3 > 2, fixing the scan domain's lower edge")
    return ok_t1 and ok_t4


def check3_rank(report):
    """rank{k_a^i} = 2  =>  2 eaten + 2 light axions (note sec 5)."""
    r = np.linalg.matrix_rank(KMAT)
    report(f"  rank of the charge matrix {{k_a^i}} = {r}  (expect 2 => 2 eaten, 2 light)")
    return r == 2


def check4_numeric_locus(report):
    """The recipe lands on the locus, inside the cone, across the scan range."""
    ok = True
    report(f"  {'t3':>7} {'t1':>11} {'t4':>11} {'max|mu|':>11} {'V_CY':>13} {'in cone':>8}")
    for t3 in T3_GRID:
        t = dflat_point(t3)
        mu = slopes(t)
        m = float(np.max(np.abs(mu)))
        in_cone = bool(np.all(t > 0))
        # slopes grow like t^2, so compare relative to the natural scale of the point
        scale = max(1.0, float(np.max(np.abs(t))) ** 2)
        good = (m / scale < TOL_SLOPE) and in_cone
        ok = ok and good
        report(f"  {t3:7.3f} {t[0]:11.4f} {t[3]:11.4f} {m:11.2e} "
               f"{volume(t):13.4f} {str(in_cone):>8}{'' if good else '   <-- FAIL'}")
    return ok


def check5_no_scale(report):
    """t^i t^j g_ij = 3/4 at every locus point.

    Consequence of V being homogeneous of degree 3; it fixes the normalisation of the
    volume-direction axion and hence the universal coupling benchmark. Same identity
    underlies the model-1 light-axion value.
    """
    ok = True
    for t3 in T3_GRID:
        t = dflat_point(t3)
        val = float(t @ axion_metric(t) @ t)
        good = abs(val - 0.75) < TOL_IDENT
        ok = ok and good
        report(f"  t3={t3:7.3f}:  t.g.t = {val:.12f}   (expect 0.75){'' if good else '  <-- FAIL'}")
    return ok


def check6_volume_is_light(report):
    """k_a^i g_ij t^j = 0 on the locus: the volume direction is always light.

    Proved in the module docstring (the overlap equals mu(L_a)/(8V)). Verified here
    numerically ON the locus, and -- as a control -- shown to be NONZERO off it, which
    confirms the test is actually sensitive to D-flatness rather than trivially true.
    """
    ok = True
    for t3 in [2.5, 3.0, 4.0, 6.0, 10.0]:
        t = dflat_point(t3)
        g = axion_metric(t)
        ov = [abs(float(KMAT[:, a] @ g @ t)) for a in range(5)]
        good = max(ov) < TOL_IDENT
        ok = ok and good
        report(f"  t3={t3:6.3f}:  max_a |k_a.g.t| = {max(ov):.3e}{'' if good else '  <-- FAIL'}")

    # Control: step OFF the D-flat locus; the overlap must become nonzero, and must
    # equal mu(L_a)/(8V) as proved.
    t_off = dflat_point(3.0)
    t_off[2] += 0.7                       # break D-flatness, stay inside the cone
    g_off = axion_metric(t_off)
    V_off = volume(t_off)
    mu_off = slopes(t_off)
    pred = mu_off / (8.0 * V_off)
    got = np.array([float(KMAT[:, a] @ g_off @ t_off) for a in range(5)])
    off_nonzero = float(np.max(np.abs(got))) > 1e-6
    pred_ok = bool(np.allclose(got, pred, rtol=1e-10, atol=1e-14))
    report(f"  control OFF the locus (t={np.round(t_off, 4)}):")
    report(f"    max_a |k_a.g.t| = {np.max(np.abs(got)):.6e}  (nonzero as required: {off_nonzero})")
    report(f"    matches the proved value mu(L_a)/(8V): {pred_ok}")
    return ok and off_nonzero and pred_ok


def main():
    lines = []

    def report(s):
        lines.append(s)
        print(s)

    checks = []
    print("=" * 78)
    print("Model 2: D-flat Kahler locus and axion-split verification")
    print("=" * 78)

    print("\nCHECK 1 -- slopes derived from scratch vs the note's quoted formulas")
    ok1, mus, ts = check1_symbolic_slopes(report)
    checks.append(("CHECK 1  symbolic slopes", ok1))

    print("\nCHECK 2 -- solve mu(L1)=mu(L4)=0 at t2=1 from scratch")
    checks.append(("CHECK 2  symbolic locus", check2_symbolic_locus(report, mus, ts)))

    print("\nCHECK 3 -- rank of the charge matrix (eaten/light count)")
    checks.append(("CHECK 3  rank => 2+2", check3_rank(report)))

    print("\nCHECK 4 -- recipe lands on the locus, inside the Kahler cone")
    checks.append(("CHECK 4  numeric locus", check4_numeric_locus(report)))

    print("\nCHECK 5 -- no-scale identity t.g.t = 3/4")
    checks.append(("CHECK 5  no-scale identity", check5_no_scale(report)))

    print("\nCHECK 6 -- volume direction is g-orthogonal to the eaten span (on-locus only)")
    checks.append(("CHECK 6  volume is light", check6_volume_is_light(report)))

    print("\n" + "=" * 78)
    for name, ok in checks:
        print(f"  {name:34s} {'PASS' if ok else 'FAIL'}")
    all_ok = all(ok for _, ok in checks)
    print("=" * 78)
    print("RESULT:", "ALL CHECKS PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
