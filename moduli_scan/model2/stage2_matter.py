#!/usr/bin/env python3
# Ported from model2_matter_fields_all.ipynb (repository root).
# The notebook cell code is VERBATIM except for lines tagged [SCAN] (the scan harness).
# Stage 2 (model 2): build every matter field's harmonic representative and bundle metric,
#          save model2_matter_fields.pkl. Runs INSIDE --run-dir so the notebook's relative
#          globs (model2_run_*.pkl -> model2_matter_fields.pkl) resolve per-run.
import argparse, os
import matplotlib
matplotlib.use('Agg')

_ap = argparse.ArgumentParser()
_ap.add_argument("--run-dir", dest="run_dir", required=True)
_ap.add_argument("--t3", type=float, required=True,
                 help="[SCAN] must equal the t3 stage 1 trained at; the notebook asserts this")
_ap.add_argument("--parity", choices=["corrected", "legacy"], default="corrected",
                 help="[SCAN] 'corrected' (default) is the notebooks' CURRENT table, taken "
                      "verbatim: eps=(-1)^(a+b+1) for L5, (-1)^(b+1) for fb. 'legacy' restores "
                      "the retired pre-2026-08-13 table whose sigma projection cannot fit, and "
                      "exists only to reproduce the parity A/B experiment. The old spelling "
                      "'notebook' was removed: after 2026-08-13 it would mean its own opposite.")
ARGS = _ap.parse_args()
os.chdir(ARGS.run_dir)  # [SCAN] per-run file handoff

import numpy as np
import os, pickle, glob, itertools, math
import jax, jax.numpy as jnp
import equinox as eqx
from cymetric.jax.models.models import PhiFSModel
from cymetric.jax.models.helper import prepare_basis as jax_prepare_basis

# projective input features
def kappa_features(z):
    xR = z[:8];  xI = z[8:]
    mag2      = xR**2 + xI**2
    pair_mag2 = mag2.reshape(4, 2).sum(axis=1)
    kap       = jnp.sqrt(jnp.repeat(pair_mag2, 2))
    xR = xR / kap;  xI = xI / kap
    a = jnp.array([0, 2, 4, 6]);  b = jnp.array([1, 3, 5, 7])
    feats = jnp.stack([xR[a]*xR[b] + xI[a]*xI[b], xR[a]*xI[b] - xI[a]*xR[b],
                       xR[a]**2 + xI[a]**2, xR[b]**2 + xI[b]**2], axis=1)
    return feats.reshape(-1)

# --- freely-acting Z2: flip the 0-coordinate of every P1 ---
_G1IDX = jnp.array([1, 3, 5, 7, 9, 11, 13, 15])          # flips x_{i,1} == flips x_{i,0} projectively
def _g1(z): return z.at[_G1IDX].multiply(-1.0)

class OrbitAvg(eqx.Module):
    net: eqx.Module
    def __call__(self, x):
        return 0.5 * ( self.net(x) + self.net(_g1(x)) )   # Z2 average (was 0.25*4 for Z2xZ2)

print("Z2 orbit-average defined (single generator _g1).")

# ------- SPECIFY THE KAHLER POINT (the one free shape modulus) -------
t2 = 1.0
t3 = ARGS.t3  # [SCAN] scan variable; must match the t3 stage 1 trained at
t1 = (t3 - 2.0) * (t3 + 1.0)              # = t3^2 - t3 - 2   (mu(L4)=0 at t2=1)
t4 = t1 / t3                              # mu(L1)=0 :  t1 t2 = t3 t4
t_locus = np.array([t1, t2, t3, t4], dtype=float)
assert np.all(t_locus > 0), f"NOT in Kahler cone: t={t_locus}"     # tetra-quadric cone = positive orthant

# verify D-flatness (all five slopes vanish)  and the volume
D3 = np.zeros((4,4,4))
for i in range(4):
    for j in range(4):
        for k in range(4):
            if len({i,j,k})==3: D3[i,j,k]=2.0
kmat = np.array([[-1,-1,-1,1,2],[-1,-1,-1,2,1],[1,1,1,-3,0],[1,1,1,-1,-2]])   # rows=P1, cols=L1..L5
mu = [float(np.einsum('ijk,i,j,k->', D3, kmat[:,a], t_locus, t_locus)) for a in range(5)]
V_CY = float((1/6)*np.einsum('ijk,i,j,k->', D3, t_locus, t_locus, t_locus))
print(f"t = {t_locus}   (t3={t3})")
print(f"slopes mu(L_a) = {np.round(mu,8)}   (all ~0 => D-flat)")
print(f"V_CY = {V_CY:.6f}    (NOT 8 -- Kahler-point dependent)")

def to_jax(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, tree)

pkl_files = sorted(glob.glob('model2_run_*.pkl'))
assert pkl_files, "No model2_run_*.pkl found -- train the metric (at kmoduli=t_locus), HYM and sigma nets first."
filename = pkl_files[-1]
print("Loading:", filename)
with open(filename, 'rb') as f:
    saved = pickle.load(f)
meta      = saved['metadata']
BASIS     = jax_prepare_basis(saved['BASIS_raw'])
psi0, psi1 = meta['psi0'], meta['psi1']
np.testing.assert_allclose(meta['kmoduli'], t_locus, rtol=1e-6,
    err_msg="metric net kmoduli != the D-flat point t_locus")     # <-- the metric MUST be at t_locus

metric_model = to_jax(saved['models']['metric_model'])
hym_models   = {k: to_jax(v) for k, v in saved['models']['hym_models'].items()}   # keys: 'L5','fb'
sigma_models = {k: to_jax(v) for k, v in saved['models']['sigma_models'].items()} # keys: the 8 harmonic-form nets

X_val         = saved['training_data']['X_val'].astype(np.float32)
y_val         = saved['training_data']['y_val'].astype(np.float32)
val_pullbacks = saved['training_data']['val_pullbacks']
N_val = X_val.shape[0]
print(f"psi0={psi0} psi1={psi1}  kmoduli={meta['kmoduli']}")
print(f"HYM: {list(hym_models)}   sigma: {list(sigma_models)}")
print(f"X_val {X_val.shape}, val_pullbacks {val_pullbacks.shape}")

K_L5 = np.array([2,1,0,-2]);  K_FB = np.array([0,1,-2,0])          # bundle charges (P1 degrees)

def _mu_bar(x_cplx, j):        # (0,1)-leg on factor j:  slots (2j,2j+1) = (-conj x_{j,1}, conj x_{j,0})
    v = jnp.zeros(8, dtype=jnp.complex64)
    v = v.at[2*j].set(-jnp.conj(x_cplx[2*j+1])); v = v.at[2*j+1].set(jnp.conj(x_cplx[2*j]))
    return v

def nu_ref_L5(x_cplx, a, b):   # L5 = O(2,1,0,-2), H^1 on factor 3
    kap3 = jnp.sum(jnp.abs(x_cplx[6:8])**2)
    poly = x_cplx[0]**(2-a) * x_cplx[1]**a * x_cplx[2]**(1-b) * x_cplx[3]**b
    return (poly / kap3**2) * _mu_bar(x_cplx, 3)

def nu_ref_FB(x_cplx, b):      # fb = O(0,1,-2,0), H^1 on factor 2
    kap2 = jnp.sum(jnp.abs(x_cplx[4:6])**2)
    poly = x_cplx[2]**(1-b) * x_cplx[3]**b
    return (poly / kap2**2) * _mu_bar(x_cplx, 2)

def nu_ref(x_cplx, spec):      # spec = ('L5',(a,b)) or ('fb',(b,))
    if spec[0]=='L5': return nu_ref_L5(x_cplx, *spec[1])
    return nu_ref_FB(x_cplx, spec[1][0])

# harmonic sigma correction, Z2-symmetrised to definite parity eps (=+1 invariant / -1 anti-invariant)
def n_sections(k):
    n=1
    for ki in k: n*=abs(ki)+1
    return n
def build_sections(x_cplx, k):    # ambient monomial basis
    per=[]
    for i,ki in enumerate(k):
        xi=x_cplx[2*i:2*i+2]; m=abs(ki)
        if ki>=0: segs=[xi[0]**(m-a)*xi[1]**a for a in range(m+1)]
        else:
            kap=jnp.sum(jnp.abs(xi)**2); segs=[jnp.conj(xi[0])**(m-a)*jnp.conj(xi[1])**a/kap**m for a in range(m+1)]
        per.append(jnp.stack(segs))
    r=per[0]
    for s in per[1:]: r=jnp.outer(r,s).reshape(-1)
    return r.astype(jnp.complex64)
def sigma_fn(model, x, k, eps):
    N=n_sections(k)
    def raw(z):
        xc=(z[:8]+1j*z[8:]).astype(jnp.complex64); s=build_sections(xc,k)
        out=model(z); c=out[:N]+1j*out[N:]; return jnp.dot(c,s)
    return 0.5*(raw(x)+eps*raw(_g1(x)))
def dbar_sigma(model, x, k, eps):
    def sR(z): return jnp.real(sigma_fn(model,z,k,eps))
    def sI(z): return jnp.imag(sigma_fn(model,z,k,eps))
    gR=jax.grad(sR)(x); gI=jax.grad(sI)(x)
    return (0.5*((gR[:8]+1j*gR[8:])+1j*(gI[:8]+1j*gI[8:]))).astype(jnp.complex64)
print("Type-I reference forms + Z2 sigma correction defined.")

# field -> (form spec, bundle key, g-parity eps)
FIELD_MAP = {
    # eps = form g1-eigenvalue: (-1)^(a+b+1) for L5, (-1)^(b+1) for fb  (u^c/e^c: -1,  Q: +1)
    'U1':(('L5',(0,0)),'L5',-1), 'U2':(('L5',(1,1)),'L5',-1), 'U3':(('L5',(2,0)),'L5',-1),
    'Q1':(('L5',(0,1)),'L5',+1), 'Q2':(('L5',(1,0)),'L5',+1), 'Q3':(('L5',(2,1)),'L5',+1),
    'E1':(('L5',(0,0)),'L5',-1), 'E2':(('L5',(1,1)),'L5',-1), 'E3':(('L5',(2,0)),'L5',-1),   # e^c = u^c
    'D' :(('fb',(0,)), 'fb',-1),                                  # d^c  (U(3)-degenerate representative)
    'L' :(('fb',(1,)), 'fb',+1),                                  # lepton doublet
}
_FIELD_MAP_LEGACY = {  # [SCAN] the RETIRED pre-2026-08-13 table; --parity legacy only
    # [SCAN] 10 : parity (-1)^(a+b).  U/E parity +1 ; Q parity -1
    'U1':(('L5',(0,0)),'L5',+1), 'U2':(('L5',(1,1)),'L5',+1), 'U3':(('L5',(2,0)),'L5',+1),  # [SCAN]
    'Q1':(('L5',(0,1)),'L5',-1), 'Q2':(('L5',(1,0)),'L5',-1), 'Q3':(('L5',(2,1)),'L5',-1),  # [SCAN]
    'E1':(('L5',(0,0)),'L5',+1), 'E2':(('L5',(1,1)),'L5',+1), 'E3':(('L5',(2,0)),'L5',+1),  # [SCAN] e^c = u^c
    'D' :(('fb',(0,)), 'fb',+1),  # [SCAN] d^c (U(3)-degenerate representative)
    'L' :(('fb',(1,)), 'fb',-1),  # [SCAN] lepton doublet
}
if ARGS.parity == "legacy":  # [SCAN] reproduce the buggy pre-fix parity, for the A/B only
    FIELD_MAP = _FIELD_MAP_LEGACY  # [SCAN]
BUND_K = {'L5':K_L5, 'fb':K_FB}
# charge sanity: L5 parity of each 10 form, fb parity of each 5bar form
for f,(spec,bnd,eps) in FIELD_MAP.items():
    if spec[0]=='L5': p=(-1)**(sum(spec[1])+1)   # g1 flips x_{i,1}: parity (-1)^(a+b+1)
    else: p=(-1)**(spec[1][0]+1)
    if ARGS.parity == "legacy": p = -p  # [SCAN] the retired table dropped the mu_bar leg's -1
    assert p==eps, (f,p,eps)
print("Fields:", list(FIELD_MAP), " -- parities consistent with the Wilson projection.")

from tqdm import tqdm
CHUNK = 512
@jax.jit
def geom_chunk(Xc, Jc):
    g = metric_model(Xc); ginv = jnp.linalg.inv(g)
    xc = (Xc[:, :8] + 1j*Xc[:, 8:]).astype(jnp.complex64)
    def gref_i(i):
        Ji = Jc[:, :, 2*i:2*i+2]; xi = xc[:, 2*i:2*i+2]
        ki = jnp.sum(jnp.abs(xi)**2, axis=1)[:, None, None]
        JiJi = jnp.einsum('...ai,...bi->...ab', Ji, jnp.conj(Ji))
        Jixi = jnp.einsum('...ai,...i->...a', Ji, xi)
        out  = jnp.einsum('...a,...b->...ab', Jixi, jnp.conj(Jixi))
        return (ki*JiJi - out) / (ki**2 * jnp.pi)
    gref = jnp.stack([gref_i(i) for i in range(4)], axis=1)
    return g, ginv, gref
@jax.jit
def pb_chunk(Xc): return metric_model.pullbacks(Xc)

Xj = jnp.array(X_val)
g = np.zeros((N_val,3,3),np.complex64); gi = np.zeros((N_val,3,3),np.complex64)
gr = np.zeros((N_val,4,3,3),np.complex64); J = np.zeros((N_val,3,8),np.complex64)
for s in tqdm(range(0, N_val, CHUNK)):
    e=min(s+CHUNK,N_val); Jb=pb_chunk(Xj[s:e])
    gg,ii,rr = geom_chunk(Xj[s:e], Jb)
    g[s:e]=np.array(gg); gi[s:e]=np.array(ii); gr[s:e]=np.array(rr); J[s:e]=np.array(Jb)
CLOUD = dict(g=g, ginv=gi, gref=gr, J=J)
print("Geometry ready (metric at t_locus).")

def build_field(spec, bnd, eps):
    k_tuple = tuple(int(v) for v in BUND_K[bnd])
    # each field's sigma net is keyed by the field name; pass it in via closure
    def nu_pb_fn(model_s, Xb, Jb):
        @jax.jit
        def single(x, J):
            xc = (x[:8] + 1j*x[8:]).astype(jnp.complex64)
            return jnp.conj(J) @ (nu_ref(xc, spec) + dbar_sigma(model_s, x, k_tuple, eps))
        return jax.vmap(single)(Xb, Jb)
    def H_fn(Xb):
        k_arr = jnp.array(BUND_K[bnd], dtype=jnp.float32); bm = hym_models[bnd]
        @jax.jit
        def single(x):
            xc=(x[:8]+1j*x[8:]).astype(jnp.complex64)
            kap=jnp.stack([jnp.sum(jnp.abs(xc[2*i:2*i+2])**2) for i in range(4)])
            return (jnp.prod(kap**(-k_arr)).real)*jnp.exp(bm(x)[0]).real
        return jax.vmap(single)(Xb)
    return nu_pb_fn, H_fn

CH = 256; nu_pb = {}; Hvals = {}
Xj = jnp.array(X_val); Jj = jnp.array(CLOUD['J'])
for fld,(spec,bnd,eps) in FIELD_MAP.items():
    model_s = sigma_models['U'+fld[1] if fld.startswith('E') else fld]   # E reuses U's sigma net
    nfn, hfn = build_field(spec, bnd, eps)
    nu = np.zeros((N_val,3),np.complex64); H = np.zeros(N_val,np.float64)
    for s in range(0, N_val, CH):
        e=min(s+CH,N_val)
        nu[s:e]=np.array(nfn(model_s, Xj[s:e], Jj[s:e])); H[s:e]=np.array(hfn(Xj[s:e]))
    nu_pb[fld]=nu; Hvals[fld]=H
    print(f"  {fld:3s} ({bnd}, {spec[1]}, eps={eps:+d}):  <|nu|>={np.abs(nu).mean():.4f}  H in [{H.min():.3f},{H.max():.3f}]")
print("All matter-field wavefunctions computed.")

SECTORS = {                                     # sector -> (fields, block-end or None, universal-x3?)
    'Q': (['Q1','Q2','Q3'], None, False),       # charge e5, genuine 3x3
    'U': (['U1','U2','U3'], None, False),        # charge e5, genuine 3x3
    'E': (['E1','E2','E3'], None, False),        # = U
    'D': (['D'],            None, True ),        # d^c, U(3): scalar x 3 (charges e_a+e4)
    'L': (['L'],            None, True ),        # lepton, U(3): scalar x 3
}
out = dict(
    nu_pb=nu_pb, Hvals=Hvals,
    cloud=dict(g=CLOUD['g'], ginv=CLOUD['ginv'], gref=CLOUD['gref']),
    field_map=FIELD_MAP, sectors=SECTORS,
    t_locus=t_locus, V_CY=V_CY,
    aux=((y_val[:,0]/y_val[:,1])/6.0),          # CY measure weight
    K_L5=K_L5, K_FB=K_FB,
)
with open('model2_matter_fields.pkl','wb') as f:
    pickle.dump(out, f)
print("Saved model2_matter_fields.pkl  (fields %d, N=%d, t=%s)" % (len(nu_pb), N_val, t_locus))
