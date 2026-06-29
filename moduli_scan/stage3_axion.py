#!/usr/bin/env python3
# Stage 3: assemble the axion-matter coupling matrices and serialize the results.
# Runs INSIDE the run dir (reads matter_fields_all.pkl) and writes results.json.
import argparse, os, json
_ap = argparse.ArgumentParser()
_ap.add_argument("--run-dir", dest="run_dir", required=True)
_ap.add_argument("--psi1", type=complex, required=True)  # complex-capable
_ap.add_argument("--seed", type=int, required=True)
ARGS = _ap.parse_args()
os.chdir(ARGS.run_dir)  # per-run file handoff

import numpy as np, pickle
print("="*66); print(" LOAD PRECOMPUTED MATTER FIELDS  (from matter_fields_all.ipynb)"); print("="*66)
with open('matter_fields_all.pkl','rb') as f:
    M = pickle.load(f)
nu_pb   = M['nu_pb']; Hvals = M['Hvals']; CLOUDS = M['clouds']
SECTORS = M['sectors']; FIELD_MAP = M['field_map']; FACTOR_PERM = M['factor_perm']
aux = M['aux']; V_CY = M['V_CY']
N_val = len(aux)
print("fields :", list(nu_pb)); print("sectors:", list(SECTORS)); print("clouds :", list(CLOUDS), " N =", N_val)

# The two overlap integrals, per sector, on that sector's cloud:
#   matter Kahler metric  G_IJ = (1/2V) INT nu_I ^ star(H nu_J)          [the NORMALISATION]
#   axion coupling        Lambda_iIJ = (1/2V) INT (star omega_i) ^ nu_I ^ (H nu_J)
# Both reduce to Levi-Civita wedge contractions over the trained geometry (eps eps g g / eps eps g W).
# G here is the matter Kahler metric G_IJ = (1/2V) INT nu_I^star(H nu_J), i.e. (1/4V)<aux*eps eps g g nu(Hnu)>
# -- the SAME convention as axion_coupling.ipynb (it is 2x the up-Yukawa notebook's K, which carries an extra 1/2).
_e3 = np.zeros((3,3,3))
for _p,_s in [((0,1,2),1),((1,2,0),1),((2,0,1),1),((0,2,1),-1),((2,1,0),-1),((1,0,2),-1)]: _e3[_p]=_s

def sector_integrals(forms, cloud_key, block_end, perm_key):
    g = CLOUDS[cloud_key]['g']; ginv = CLOUDS[cloud_key]['ginv']; gref = CLOUDS[cloud_key]['gref']
    trgW = np.real(np.einsum('nab,niba->ni', ginv, gref))          # Lambda(omega_i)=tr(g^{-1}W^{(i)})
    nu = np.stack([nu_pb[f] for f in forms], 0)
    n = len(forms); G = np.zeros((n,n),complex); Lam = np.zeros((4,n,n),complex)
    for I in range(n):
        H_I = Hvals[forms[I]]
        for J in range(n):
            Hnu = H_I[:,None]*np.conj(nu[J])
            Qgg = np.einsum('xab,xcd,xe,xf,acf,bde->x', g, g, nu[I], Hnu, _e3, _e3)
            G[I,J] = (1.0/(4*V_CY))*np.mean(aux*Qgg)
            for i in range(4):
                QgW = np.einsum('xab,xcd,xe,xf,acf,bde->x', g, gref[:,i], nu[I], Hnu, _e3, _e3)
                Lam[i,I,J] = (-(1.0/(2*V_CY))*np.mean(aux*QgW) + (1.0/(4*V_CY))*np.mean(aux*trgW[:,i]*Qgg))
    G = 0.5*(G+G.conj().T)
    for i in range(4): Lam[i] = 0.5*(Lam[i]+Lam[i].conj().T)
    # relabel Lambda's modulus index back to the unpermuted basis: Lambda^F_i = Lambda^computed_{sigma(i)}
    fp = FACTOR_PERM[perm_key]; Lam = Lam[fp, :, :]
    if block_end is not None:                                       # S(U(1)^5): L_2 block decouples from L_5
        G[:block_end, block_end:] = G[block_end:, :block_end] = 0
        for i in range(4): Lam[i,:block_end,block_end:] = Lam[i,block_end:,:block_end] = 0
    return G, Lam

results = {}
for r,(forms,cloud,blk) in SECTORS.items():
    perm_key = FIELD_MAP[forms[0]][1]
    G, Lam = sector_integrals(forms, cloud, blk, perm_key)
    results[r] = dict(G=G, Lam=Lam, forms=forms)
    print(f"sector {r:6s} {forms}: computed G ({G.shape}) and Lambda")

# Matter Kahler metrics (normalisations) for every sector -- FULL matrices (the off-diagonals are the
# flavour overlaps), and the S4 / Lambda^2 equalities.
def eig(M): return np.round(np.linalg.eigvalsh(0.5*(M+M.conj().T)),6)
def showM(M, ind="    "): print(ind + np.array2string(np.round(M,5), prefix=ind))
print("="*70); print(" MATTER KAHLER METRICS  G  (full matrices; the field normalisations)"); print("="*70)
for r in results:
    G = results[r]['G']
    print(f"\n  G^{r}  (forms {results[r]['forms']}):"); showM(G)
    if G.shape[0] > 1: print(f"    eigs = {eig(G)}")

print("\n" + "-"*70)
print(" S4 / group-theory consistency  (full matrices; residual = trained-net S4 breaking)")
print("-"*70)
print(f"  H^d = {float(np.real(results['Hd']['G'][0,0])):.6f}   vs   H^u = {float(np.real(results['H']['G'][0,0])):.6f}")
print("  D (L4L5) full G  vs  Q (L_e2) 2x2 block:"); showM(results['D_4_5']['G']); showM(results['Q']['G'][:2,:2])
print("  L (L4L5) full G  vs  U (L_e2) 2x2 block:"); showM(results['L_4_5']['G']); showM(results['U']['G'][:2,:2])
print(f"  D(L2L4) = {float(np.real(results['D_2_4']['G'][0,0])):.6f} vs Q3 = {float(np.real(results['Q']['G'][2,2])):.6f}"
      f"    L(L2L4) = {float(np.real(results['L_2_4']['G'][0,0])):.6f} vs U3 = {float(np.real(results['U']['G'][2,2])):.6f}")
print("  E full G  vs  U full G  (e^c = u^c, exact):"); showM(results['E']['G']); showM(results['U']['G'])

t_locus = np.array([1.,1.,1.,1.])
def inv_sqrt(Mx):
    ev,U = np.linalg.eigh(0.5*(Mx+Mx.conj().T)); return U @ np.diag(1.0/np.sqrt(ev)) @ U.conj().T
# print("Dilation check  D = G^{-1/2}(sum_i t^i Lambda_i)G^{-1/2}  (-> identity):\n")
# for r in results:
#     G = results[r]['G']; Lam = results[r]['Lam']
#     D = inv_sqrt(G) @ np.einsum('i,iIJ->IJ', t_locus, Lam) @ inv_sqrt(G)
#     print(f"  {r:6s} {results[r]['forms']}:  max|D-1| = {np.abs(D-np.eye(len(G))).max():.3f}")

# Axion kinetic metric on Kahler-moduli space: g_ij = (1/4) d_i d_j(-ln V), V=(1/6)d_ijk t^i t^j t^k.
def volume(t): return 2.0*(t[0]*t[1]*t[2]+t[0]*t[1]*t[3]+t[0]*t[2]*t[3]+t[1]*t[2]*t[3])
D3 = np.zeros((4,4,4))
for i in range(4):
    for j in range(4):
        for k in range(4):
            if len({i,j,k})==3: D3[i,j,k]=2.0
t = t_locus; V = volume(t)
Vi = np.einsum('ijk,j,k->i', D3, t, t)/2.0; Vij = np.einsum('ijk,k->ij', D3, t)
g_axion = 0.25*(-Vij/V + np.outer(Vi,Vi)/V**2)
print(f"V={V:.3f}; axion kinetic metric g_ij:\n{np.round(g_axion,5)}")

# eaten (anomalous U(1)) vs light (volume) axion directions
k_matrix = np.array([[-1,-1,0,1,1],[0,-3,1,1,1],[0,2,-1,-1,0],[1,2,0,-1,-2]],dtype=float)
light = np.array([1.,1.,1.,1.]); light = light/np.sqrt(light@g_axion@light)
def g_orth(vecs, metric, against):
    out=[]
    for v in vecs:
        v = v - (v@metric@against)/(against@metric@against)*against
        for u in out: v = v - (v@metric@u)/(u@metric@u)*u
        if np.sqrt(abs(v@metric@v))>1e-8: out.append(v/np.sqrt(abs(v@metric@v)))
    return out
eaten = g_orth(list(k_matrix.T), g_axion, light)
print(f"# eaten dirs = {len(eaten)} (expect 3 anomalous U(1)s)")

# Canonical axion-matter couplings  xi = e^i G^{-1/2}(i Lambda_i)G^{-1/2}, M = -i*xi (Hermitian),
# for EVERY sector (the L_2 family block, the L_5 singlet, and each Higgs/lepton singlet).
def canon(r, direction):
    G = results[r]['G']; Lam = results[r]['Lam']
    Gis = inv_sqrt(G); return -1j*(Gis @ (1j*np.einsum('i,iIJ->IJ', direction, Lam)) @ Gis)
def show(M):
    # M = -i*xi is Hermitian: diagonal = flavour-diagonal coupling, off-diagonal = flavour-changing (FCNC)
    print("        diag =", np.round(np.real(np.diag(M)),4))
    if M.shape[0] > 1:
        print("        " + np.array2string(np.round(M,4), prefix="        "))
print("LIGHT (volume) axion -- full canonical coupling M = -i*xi per sector:")
for r in results:
    print(f"  {r:6s} {results[r]['forms']}:"); show(canon(r, light))
print("\nHEAVY (eaten) axions -- full non-universal M = -i*xi (diagonal + off-diagonal FCNC) per sector:")
for r in results:
    print(f"  sector {r:6s} {results[r]['forms']}:")
    for a,e in enumerate(eaten):
        print(f"    eaten[{a}]:"); show(canon(r, e))

# ====================== serialize results ======================
# Persist, per sector/field, the canonical axion-matter coupling M=-i*xi
# diagonal (the physical per-field coupling) for the light (volume) axion and
# each eaten axion, plus the full complex G and Lambda for completeness.
def _c2list(a):
    a = np.asarray(a)
    return {"re": np.real(a).tolist(), "im": np.imag(a).tolist()}

_axes = [("light", light)] + [(f"eaten{a}", e) for a, e in enumerate(eaten)]
_out = {
    "psi1": float(np.real(ARGS.psi1)),       # real part (legacy field; back-compat w/ real-scan tooling)
    "psi1_re": float(np.real(ARGS.psi1)),    # complex scan: real part
    "psi1_im": float(np.imag(ARGS.psi1)),    # complex scan: imaginary part
    "seed": int(ARGS.seed),
    "V_CY": float(V_CY),
    "axion_kinetic_metric": g_axion.tolist(),
    "n_eaten": len(eaten),
    "sectors": {},
}
for r in results:
    forms = results[r]["forms"]
    G = results[r]["G"]; Lam = results[r]["Lam"]
    Mdiag = {}
    for nm, direction in _axes:
        M = canon(r, direction)               # Hermitian; diag = per-field coupling
        Mdiag[nm] = np.real(np.diag(M)).tolist()
    _out["sectors"][r] = {
        "forms": forms,
        "G": _c2list(G),
        "Lambda": _c2list(Lam),
        "coupling_diag": Mdiag,               # {axion_name: [per-field real coupling]}
    }

with open("results.json", "w") as _f:
    json.dump(_out, _f, indent=2)
print(f"wrote {os.path.join(ARGS.run_dir, 'results.json')}  "
      f"(sectors={list(_out['sectors'])}, axes={[a for a,_ in _axes]})")

