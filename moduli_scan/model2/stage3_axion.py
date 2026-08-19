#!/usr/bin/env python3
# Ported from model2_axion_coupling_all.ipynb (repository root).
# The notebook cell code is VERBATIM except for lines tagged [SCAN];
# the [SCAN] serialization footer is appended after it.
# Stage 3 (model 2): matter Kahler metric G, coupling Lambda_i, and the canonical
#          axion couplings at the D-flat point; writes results.json into --run-dir.
import argparse, os, json
_ap = argparse.ArgumentParser()
_ap.add_argument("--run-dir", dest="run_dir", required=True)
_ap.add_argument("--t3", type=float, required=True)
_ap.add_argument("--seed", type=int, required=True)
# [SCAN] training provenance, recorded into results.json so that undertrained smoke or
# calibration runs can never be silently averaged into a production scan by aggregate.py.
_ap.add_argument("--n-points", dest="n_points", type=int, default=-1)
_ap.add_argument("--epochs", type=int, default=-1)
_ap.add_argument("--parity", default="corrected")
ARGS = _ap.parse_args()
os.chdir(ARGS.run_dir)  # [SCAN] per-run file handoff

import numpy as np, pickle
with open('model2_matter_fields.pkl','rb') as f:
    M = pickle.load(f)
nu_pb = M['nu_pb']; Hvals = M['Hvals']; CLOUD = M['cloud']
SECTORS = M['sectors']; FIELD_MAP = M['field_map']
aux = M['aux']; V_CY = M['V_CY']; t_locus = np.asarray(M['t_locus'])
N_val = len(aux)
print("fields :", list(nu_pb))
print("sectors:", list(SECTORS))
print("t_locus:", t_locus, "  V_CY:", round(V_CY,4), "  N:", N_val)

_e3 = np.zeros((3,3,3))
for _p,_s in [((0,1,2),1),((1,2,0),1),((2,0,1),1),((0,2,1),-1),((2,1,0),-1),((1,0,2),-1)]: _e3[_p]=_s
g = CLOUD['g']; ginv = CLOUD['ginv']; gref = CLOUD['gref']
trgW = np.real(np.einsum('nab,niba->ni', ginv, gref))          # Lambda(omega_i)=tr(g^{-1}W^{(i)})

def sector_integrals(fields):
    nu = np.stack([nu_pb[f] for f in fields], 0)
    n = len(fields); G = np.zeros((n,n),complex); Lam = np.zeros((4,n,n),complex)
    for I in range(n):
        H_I = Hvals[fields[I]]
        for J in range(n):
            Hnu = H_I[:,None]*np.conj(nu[J])
            Qgg = np.einsum('xab,xcd,xe,xf,acf,bde->x', g, g, nu[I], Hnu, _e3, _e3)
            G[I,J] = (1.0/(4*V_CY))*np.mean(aux*Qgg)
            for i in range(4):
                QgW = np.einsum('xab,xcd,xe,xf,acf,bde->x', g, gref[:,i], nu[I], Hnu, _e3, _e3)
                # 0.5 = GS/one-loop vertex 1/(4pi)
                Lam[i,I,J] = 0.5*(-(1.0/(2*V_CY))*np.mean(aux*QgW) + (1.0/(4*V_CY))*np.mean(aux*trgW[:,i]*Qgg))
    G = 0.5*(G+G.conj().T)
    for i in range(4): Lam[i] = 0.5*(Lam[i]+Lam[i].conj().T)
    return G, Lam

results = {}
for r,(fields, blk, univ) in SECTORS.items():
    G, Lam = sector_integrals(fields)
    results[r] = dict(G=G, Lam=Lam, fields=fields, univ=univ)
    print(f"sector {r:2s} {fields}: G {G.shape}{'  (U(3): scalar x3)' if univ else ''}")

def showM(M, ind="    "): print(ind + np.array2string(np.round(M,5), prefix=ind))
def eig(M): return np.round(np.linalg.eigvalsh(0.5*(M+M.conj().T)),6)
for r in results:
    G = results[r]['G']
    print(f"\n  G^{r}  {results[r]['fields']}{'  x3 (U(3))' if results[r]['univ'] else ''}:"); showM(G)
    if G.shape[0] > 1: print(f"    eigs = {eig(G)}")

def volume(t): return 2.0*(t[0]*t[1]*t[2]+t[0]*t[1]*t[3]+t[0]*t[2]*t[3]+t[1]*t[2]*t[3])
D3 = np.zeros((4,4,4))
for i in range(4):
    for j in range(4):
        for k in range(4):
            if len({i,j,k})==3: D3[i,j,k]=2.0
t = t_locus; V = volume(t)
Vi = np.einsum('ijk,j,k->i', D3, t, t)/2.0; Vij = np.einsum('ijk,k->ij', D3, t)
g_axion = 0.25*(-Vij/V + np.outer(Vi,Vi)/V**2)          # (1/4) dd(-ln V) at t_locus
print("g_axion at t_locus:\n", np.round(g_axion,5))

def g_orth(vecs, metric, prev=None):
    out = list(prev) if prev else []
    for v in vecs:
        v = np.asarray(v, float).copy()
        for u in out: v = v - (v@metric@u)/(u@metric@u)*u
        if np.sqrt(abs(v@metric@v)) > 1e-9: out.append(v/np.sqrt(v@metric@v))
    return out

k1 = np.array([-1.,-1,1,1]); k4 = np.array([1.,2,-3,-1])          # the 2 independent anomalous dirs
eaten = g_orth([k1, k4], g_axion)                                 # 2 eaten (g-orthonormal)
light = g_orth([np.eye(4)[i] for i in range(4)], g_axion, prev=eaten)[len(eaten):]   # 2 light
vol_dir = t / np.sqrt(t @ g_axion @ t)                            # volume/radial direction (unit in g)
print(f"# eaten = {len(eaten)}   # light = {len(light)}   (expect 2 + 2)")
print("volume direction in light space?  |g-overlap with eaten| =",
      np.round([abs(vol_dir@g_axion@e) for e in eaten],6))

def canon(r, direction):
    G = results[r]['G']; Lam = results[r]['Lam']
    ev,U = np.linalg.eigh(0.5*(G+G.conj().T)); Gis = U @ np.diag(1/np.sqrt(ev)) @ U.conj().T
    xi = -1j*(Gis @ (1j*np.einsum('i,iIJ->IJ', direction, Lam)) @ Gis)
    return (1/np.sqrt(2)) * xi                          # 1/sqrt2: canonical REAL axion (not the complex modulus)

def show(M):
    print("        diag =", np.round(np.real(np.diag(M)),4))
    if M.shape[0] > 1: print("        " + np.array2string(np.round(M,4), prefix="        "))

print("="*70); print(" VOLUME-DIRECTION axion  (universal check: prop identity per sector)"); print("="*70)
for r in results:
    print(f"  {r:2s} {results[r]['fields']}:"); show(canon(r, vol_dir))

print("\n" + "="*70); print(" TWO DIAGONALISED LIGHT axions  (the physical output)"); print("="*70)
for r in results:
    print(f"  sector {r:2s} {results[r]['fields']}:")
    for a,e in enumerate(light):
        tag = "  (~volume, universal)" if abs(abs(e@g_axion@vol_dir)-1) < 1e-3 else ""
        print(f"    light[{a}]{tag}:"); show(canon(r, e))

# ============================ [SCAN] serialization ============================
import numpy as _np

def _c2list(a):
    a = _np.asarray(a)
    return {"re": _np.real(a).tolist(), "im": _np.imag(a).tolist()}

# Canonical, t3-stable light basis (see the builder's note): volume + shape.
_gax = _np.asarray(g_axion, dtype=float)
_t = _np.asarray(t_locus, dtype=float)
_vol = _t / _np.sqrt(_t @ _gax @ _t)                      # unit in g_ij
_light_raw = [_np.asarray(e, dtype=float) for e in light]

# shape = the component of the light space g-orthogonal to the volume direction
_shape = None
for _e in _light_raw:
    _c = _e - (_e @ _gax @ _vol) * _vol
    _n = float(_np.sqrt(abs(_c @ _gax @ _c)))
    if _n > 1e-8:
        _shape = _c / _n
        break
if _shape is None:
    raise SystemExit("[SCAN] could not construct a shape direction g-orthogonal to volume")

# Sign convention so the shape direction does not flip between adjacent scan points:
# fix the sign by the (t3-monotone) first component.
if _shape[0] < 0:
    _shape = -_shape

# Diagnostics that must hold on the D-flat locus (cheap, and they gate the physics).
_vol_eaten_overlap = [float(abs(_vol @ _gax @ _np.asarray(_e, float))) for _e in eaten]
_shape_eaten_overlap = [float(abs(_shape @ _gax @ _np.asarray(_e, float))) for _e in eaten]
_no_scale = float(_t @ _gax @ _t)

_axes = [("volume", _vol), ("shape", _shape)]
_axes += [(f"light_raw{_a}", _e) for _a, _e in enumerate(_light_raw)]
_axes += [(f"eaten{_a}", _np.asarray(_e, float)) for _a, _e in enumerate(eaten)]

_out = {
    "model": "model2_cicy7862_num2",
    "t3": float(ARGS.t3),
    "t_locus": _np.asarray(t_locus, dtype=float).tolist(),
    "seed": int(ARGS.seed),
    "n_points": int(ARGS.n_points),
    "epochs": int(ARGS.epochs),
    "parity": str(ARGS.parity),
    # [SCAN] Which MSSM naming the sector keys below follow. From 2026-08-14 the scan
    # adopts the corrected notebooks' labels. Runs produced before that (the 50 runs of
    # the original grid) carry NO such key and follow the legacy naming, in which legacy
    # Q holds the u^c/e^c forms and legacy L holds d^c. aggregate.py keys off
    # the absence of this field to remap them; do not backfill it into old results.json.
    "label_convention": "mssm-2026-08-13",
    "V_CY": float(V_CY),
    # psi0=2, psi1=1 are FIXED for model 2 -- the complex structure is not scanned.
    "psi0": 2.0,
    "psi1": 1.0,
    # Both factors of the 1/(2 sqrt 2) split are already baked into the stored numbers:
    # the 0.5 (GS/one-loop vertex 1/(4pi)) sits inside Lambda (notebook cell c3) and the
    # 1/sqrt(2) (real-scalar canonical normalisation) inside canon() (cell c9). Same
    # marker semantics as model 1; see moduli_scan/model1/fcnc_lib.py.
    "coupling_normalisation": "canonical",
    "axion_kinetic_metric": _gax.tolist(),
    "n_eaten": int(len(eaten)),
    "n_light": int(len(_light_raw)),
    "directions": {_nm: _v.tolist() for _nm, _v in _axes},
    "diagnostics": {
        "no_scale_t_g_t": _no_scale,                 # must be 0.75
        "vol_dot_eaten": _vol_eaten_overlap,         # must be ~0 on the D-flat locus
        "shape_dot_eaten": _shape_eaten_overlap,     # must be ~0
    },
    "sectors": {},
}

for _r in results:
    _G = results[_r]["G"]
    _Lam = results[_r]["Lam"]
    _sec = {
        "fields": list(results[_r]["fields"]),
        "univ": bool(results[_r]["univ"]),
        "G": _c2list(_G),
        "Lambda": _c2list(_Lam),
        "coupling_diag": {},
        "coupling_full": {},
    }
    for _nm, _dirv in _axes:
        _M = canon(_r, _dirv)
        _sec["coupling_diag"][_nm] = _np.real(_np.diag(_M)).tolist()
        _sec["coupling_full"][_nm] = _c2list(_M)
    _out["sectors"][_r] = _sec

with open("results.json", "w") as _f:
    json.dump(_out, _f, indent=2)
print(f"[SCAN] wrote {os.path.join(ARGS.run_dir, 'results.json')}  "
      f"(t3={ARGS.t3}, sectors={list(_out['sectors'])}, axes={[a for a,_ in _axes]})")
print(f"[SCAN] diagnostics: t.g.t={_no_scale:.12f} (expect 0.75); "
      f"max|vol.g.eaten|={max(_vol_eaten_overlap):.2e}; "
      f"max|shape.g.eaten|={max(_shape_eaten_overlap):.2e}")

