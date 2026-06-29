#!/usr/bin/env python3
# Stage 2: build the matter-field harmonic forms for every sector from the trained
# networks. Runs INSIDE the run dir so its relative globs
# (yukawa_run_*.pkl -> matter_fields_all.pkl) resolve per-run.
import argparse, os
import matplotlib
matplotlib.use('Agg')

_ap = argparse.ArgumentParser()
_ap.add_argument("--run-dir", dest="run_dir", required=True)
ARGS = _ap.parse_args()
os.chdir(ARGS.run_dir)  # per-run file handoff

import numpy as np
import os, pickle, glob, itertools, math
import jax, jax.numpy as jnp, jax.nn as jnn
import equinox as eqx
from tqdm import tqdm
from cymetric.jax.models.models import PhiFSModel
from cymetric.jax.models.helper import prepare_basis as jax_prepare_basis

# Defined before unpickling so Equinox can reconstruct the trained modules.
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

_G1IDX  = jnp.array([1, 3, 5, 7, 9, 11, 13, 15])
_G2PERM = jnp.array([1,0,3,2,5,4,7,6, 9,8,11,10,13,12,15,14])
def _g1(z): return z.at[_G1IDX].multiply(-1.0)
def _g2(z): return z[_G2PERM]

class OrbitAvg(eqx.Module):
    net: eqx.Module
    def __call__(self, x):
        return 0.25 * ( self.net(x) + self.net(_g1(x)) + self.net(_g2(x)) + self.net(_g1(_g2(x))) )

pkl_files = sorted(glob.glob('yukawa_run_*.pkl'))
assert pkl_files, "No yukawa_run_*.pkl found in this directory."
filename = pkl_files[-1]
print("="*66); print(" LOAD TRAINED RUN  (metric phi, HYM beta, harmonic-form sigma nets)"); print("="*66)
print(f"Loading: {filename}")
with open(filename, 'rb') as f:
    saved = pickle.load(f)
meta      = saved['metadata']
BASIS_raw = saved['BASIS_raw']
BASIS     = jax_prepare_basis(BASIS_raw)
print(f"Moduli    : psi0={meta['psi0']}, psi1={meta['psi1']}, kmoduli={meta['kmoduli']}")
print(f"HYM names : {meta['hym_names']}")
print(f"Form names: {meta['all_form_names']}")

def to_jax(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, tree)

psi0, psi1     = meta['psi0'], meta['psi1']
kmoduli        = np.array(meta['kmoduli'])
k_e2, k_e5, k_neg = np.array(meta['k_e2']), np.array(meta['k_e5']), np.array(meta['k_neg'])
all_form_names = meta['all_form_names']
form_names_I   = meta['form_names_I']
form_charges   = {k: np.array(v) for k, v in meta['form_charges'].items()}
combinations   = list(itertools.product([0, 1], repeat=4))

metric_model = to_jax(saved['models']['metric_model'])
hym_models   = {k: to_jax(v) for k, v in saved['models']['hym_models'].items()}
sigma_models = {k: to_jax(v) for k, v in saved['models']['sigma_models'].items()}

X_val         = saved['training_data']['X_val'].astype(np.float32)
y_val         = saved['training_data']['y_val'].astype(np.float32)
val_pullbacks = saved['training_data']['val_pullbacks']
print(f"X_val: {X_val.shape},  y_val: {y_val.shape},  val_pullbacks: {val_pullbacks.shape}")
print(f"bundles: e2={k_e2.astype(int)} e5={k_e5.astype(int)} neg={k_neg.astype(int)}")
_test = metric_model(jnp.array(X_val[:4]))
assert _test.shape == (4, 3, 3)
print("Metric model forward pass OK")

def build_sections(x_cplx, k_tuple):
    per_factor = []
    for i, ki in enumerate(k_tuple):
        xi = x_cplx[2*i:2*i+2]; abs_ki = abs(ki)
        if ki >= 0:
            segs = [xi[0]**(abs_ki - a) * xi[1]**a for a in range(abs_ki + 1)]
        else:
            kappa_i = jnp.sum(jnp.abs(xi)**2)
            segs = [jnp.conj(xi[0])**(abs_ki - a) * jnp.conj(xi[1])**a / kappa_i**abs_ki for a in range(abs_ki + 1)]
        per_factor.append(jnp.stack(segs))
    result = per_factor[0]
    for s in per_factor[1:]: result = jnp.outer(result, s).reshape(-1)
    return result.astype(jnp.complex64)

def n_sections(k_tuple):
    n = 1
    for ki in k_tuple: n *= abs(ki) + 1
    return n

form_eps = {'H': (1,-1), 'Q3': (1,-1), 'Q1': (1,-1), 'Q2': (1,-1), 'U3': (1,1), 'U1': (1,1), 'U2': (1,1)}

def sigma_fn(model, x, k_tuple, eps=(1, 1)):
    N = n_sections(k_tuple); e1, e2 = eps
    def raw(z):
        x_c = (z[:8] + 1j * z[8:]).astype(jnp.complex64)
        s   = build_sections(x_c, k_tuple)
        out = model(z); c = out[:N] + 1j * out[N:]
        return jnp.dot(c, s)
    return 0.25 * (raw(x) + e1*raw(_g1(x)) + e2*raw(_g2(x)) + e1*e2*raw(_g1(_g2(x))))

def dbar_sigma(model, x, k_tuple, eps=(1, 1)):
    def sR(z): return jnp.real(sigma_fn(model, z, k_tuple, eps))
    def sI(z): return jnp.imag(sigma_fn(model, z, k_tuple, eps))
    gR = jax.grad(sR)(x); gI = jax.grad(sI)(x)
    return (0.5 * ((gR[:8] + 1j*gR[8:]) + 1j*(gI[:8] + 1j*gI[8:]))).astype(jnp.complex64)

def nu_ref_typeI(x_cplx, form_name):
    kappas = jnp.array([jnp.sum(jnp.abs(x_cplx[2*i:2*i+2])**2) for i in range(4)])
    def mu_bar_i(i):
        v = jnp.zeros(8, dtype=jnp.complex64)
        v = v.at[2*i].set(-jnp.conj(x_cplx[2*i+1])); v = v.at[2*i+1].set(jnp.conj(x_cplx[2*i]))
        return v
    if form_name == 'H':  return (x_cplx[2]*x_cplx[3] / kappas[2]**2) * mu_bar_i(2), k_neg
    if form_name == 'U3': return ((x_cplx[2]*x_cplx[1] - x_cplx[3]*x_cplx[0]) / kappas[3]**2) * mu_bar_i(3), k_e5
    if form_name == 'Q3': return ((x_cplx[2]*x_cplx[1] + x_cplx[3]*x_cplx[0]) / kappas[3]**2) * mu_bar_i(3), k_e5
    raise ValueError(form_name)

def nu_ref_typeII(x_cplx, form_name):
    kappas = jnp.array([jnp.sum(jnp.abs(x_cplx[2*i:2*i+2])**2) for i in range(4)])
    k1, k2 = kappas[0], kappas[1]
    x10, x11, x20, x21 = x_cplx[0], x_cplx[1], x_cplx[2], x_cplx[3]
    if   form_name == 'U1': r0 = jnp.conj(x20)**3;                r1 = jnp.conj(x21)**3
    elif form_name == 'U2': r0 = jnp.conj(x21)**2*jnp.conj(x20);  r1 = jnp.conj(x21)*jnp.conj(x20)**2
    elif form_name == 'Q1': r0 = -jnp.conj(x20)**3;               r1 = jnp.conj(x21)**3
    elif form_name == 'Q2': r0 = -jnp.conj(x21)**2*jnp.conj(x20); r1 = jnp.conj(x21)*jnp.conj(x20)**2
    else: raise ValueError(form_name)
    def p_at_x1(a0, a1):
        xs = jnp.array([a0, a1, x_cplx[2], x_cplx[3], x_cplx[4], x_cplx[5], x_cplx[6], x_cplx[7]])
        X_, Y_, U_, V_ = xs[0:2], xs[2:4], xs[4:6], xs[6:8]
        s = psi1*(X_[0]*Y_[0]*U_[0]*V_[0])*(X_[1]*Y_[1]*U_[1]*V_[1])
        for c in combinations:
            coeff = 1. if sum(c)%2==0 else psi0
            s = s + coeff*(X_[c[0]]*Y_[c[1]]*U_[c[2]]*V_[c[3]])**2.
        return s
    p0 = p_at_x1(1.+0j, 0.+0j); p2 = p_at_x1(0.+0j, 1.+0j); p1 = p_at_x1(1.+0j, 1.+0j) - p0 - p2
    xb0, xb1 = jnp.conj(x10), jnp.conj(x11)
    R_cR = (-0.5*(p1*r0+p2*r1)*jnp.abs(x10)**2*xb0 + p0*r0*jnp.abs(x10)**2*xb1 + 0.5*p0*r1*x10*xb1**2
            - 0.5*p2*r0*xb0**2*x11 - p2*r1*xb0*jnp.abs(x11)**2 + 0.5*(p0*r0+p1*r1)*xb1*jnp.abs(x11)**2)
    mu2 = jnp.zeros(8, dtype=jnp.complex64)
    mu2 = mu2.at[2].set(-jnp.conj(x_cplx[3])); mu2 = mu2.at[3].set(jnp.conj(x_cplx[2]))
    return R_cR/(k1**2*k2**5)*mu2, k_e2

def nu_ref(x_cplx, form_name):
    return nu_ref_typeI(x_cplx, form_name) if form_name in form_names_I else nu_ref_typeII(x_cplx, form_name)
print("Reference-form / sigma helpers defined.")

# --- S4 permutation map: each NEW field is the permutation-image of an up-sector field ---
# The tetra-quadric p is fully S4-symmetric (permuting the four P^1) and Gamma=Z2xZ2 acts diagonally,
# so a new bundle's reference form, HYM metric AND harmonic sigma are ALL the partner's, evaluated at
# sigma-permuted coordinates. We realise this by running the partner's trained pipeline on a permuted
# point cloud.
def perm_idx(factor_perm):                 # new factor i takes old factor factor_perm[i]
    src = []
    for i in range(4): src += [2*factor_perm[i], 2*factor_perm[i]+1]
    return np.array(src + [8+s for s in src])
PERMS = {'id': perm_idx([0,1,2,3]),
         '23': perm_idx([0,2,1,3]),        # (2 3): swap 0-indexed factors 1,2
         '1324': perm_idx([2,3,0,1])}      # (1 3)(2 4): swap (0,2) and (1,3)
FACTOR_PERM = {'id':[0,1,2,3], '23':[0,2,1,3], '1324':[2,3,0,1]}   # sigma(i), for relabelling Lambda's i-index
def apply_perm(X, key): return X[:, PERMS[key]]

# physical field -> (partner up-sector form, permutation key, bundle, sector label)
# Up-sector (cloud 'id'); down-Higgs (cloud '23'); 5bar L,D doublets (cloud '1324'); E sector reuses the up-sector form ('id').
FIELD_MAP = {
    'H':  ('H','id','neg','H'),
    'Q1': ('Q1','id','e2','Q'), 'Q2': ('Q2','id','e2','Q'), 'Q3': ('Q3','id','e5','Q'),
    'U1': ('U1','id','e2','U'), 'U2': ('U2','id','e2','U'), 'U3': ('U3','id','e5','U'),
    'Hd': ('H','23','neg','Hd'),                                              # L_2 L_5 = (2 3).H^u
    'L1': ('U1','1324','e2','L_4_5'), 'L2': ('U2','1324','e2','L_4_5'),       # L on L_4 L_5 = (1 3)(2 4).U
    'D1': ('Q1','1324','e2','D_4_5'), 'D2': ('Q2','1324','e2','D_4_5'),       # D on L_4 L_5 = (1 3)(2 4).Q
    'L3': ('U3','1324','e5','L_2_4'), 'D3': ('Q3','1324','e5','D_2_4'),       # (L,D) on L_2 L_4
    'E1': ('U1','id','e2','E'), 'E2': ('U2','id','e2','E'), 'E3': ('U3','id','e5','E'),  # e^c = u^c (group theory)
}
# bundle charge of the NEW field (sigma . partner charge) -- for the record / documentation
def perm_charge(kv, key):
    fp = FACTOR_PERM[key]; return np.array([kv[fp[i]] for i in range(4)])
print("Fields:", list(FIELD_MAP))
print("New bundles (charge after permutation):")
for f in ['Hd','L1','D1','L3']:
    part, key, bnd, sec = FIELD_MAP[f]
    print(f"  {f:3s} <- {part:2s} via {key:5s}: charge {perm_charge({'neg':k_neg,'e2':k_e2,'e5':k_e5}[bnd].astype(int), key)}  (sector {sec})")

# Geometry (g, g^{-1}, the four reference (1,1)-forms omega_i) on each ORBIT CLOUD we need.
CHUNK = 512
@jax.jit
def geom_chunk(Xc, Jc):
    g = metric_model(Xc); ginv = jnp.linalg.inv(g)
    xc = (Xc[:, :8] + 1j * Xc[:, 8:]).astype(jnp.complex64)
    def gref_i(i):
        Ji = Jc[:, :, 2*i:2*i+2]; xi = xc[:, 2*i:2*i+2]
        ki = jnp.sum(jnp.abs(xi)**2, axis=1)[:, None, None]
        JiJi = jnp.einsum('...ai,...bi->...ab', Ji, jnp.conj(Ji))
        Jixi = jnp.einsum('...ai,...i->...a', Ji, xi)
        out  = jnp.einsum('...a,...b->...ab', Jixi, jnp.conj(Jixi))
        return (ki * JiJi - out) / (ki**2 * jnp.pi)
    gref = jnp.stack([gref_i(i) for i in range(4)], axis=1)
    return g, ginv, gref

@jax.jit
def pb_chunk(Xc): return metric_model.pullbacks(Xc)

N_val = X_val.shape[0]
CLOUDS = {}
for key in ['id', '23', '1324']:
    Xc = apply_perm(X_val, key); Xj = jnp.array(Xc)
    g = np.zeros((N_val,3,3),np.complex64); gi = np.zeros((N_val,3,3),np.complex64)
    gr = np.zeros((N_val,4,3,3),np.complex64); J = np.zeros((N_val,3,8),np.complex64)
    print(f"cloud '{key}': geometry + pullbacks ...")
    for s in tqdm(range(0, N_val, CHUNK)):
        e = min(s+CHUNK, N_val); Jb = pb_chunk(Xj[s:e])
        gg, ii, rr = geom_chunk(Xj[s:e], Jb)
        g[s:e]=np.array(gg); gi[s:e]=np.array(ii); gr[s:e]=np.array(rr); J[s:e]=np.array(Jb)
    CLOUDS[key] = dict(X=Xc, J=J, g=g, ginv=gi, gref=gr)
print("Geometry ready on clouds:", list(CLOUDS))

# Corrected harmonic form (nu_ref + dbar sigma) and bundle metric H, for EVERY physical field,
# each on its own orbit cloud (the partner's trained nets do all the work).
def make_nu_pb_fn(partner, k_tuple):
    model_s = sigma_models[partner]
    @jax.jit
    def fn(Xb, Jb):
        def single(x, J):
            x_c = (x[:8] + 1j * x[8:]).astype(jnp.complex64)
            return jnp.conj(J) @ (nu_ref(x_c, partner)[0] + dbar_sigma(model_s, x, k_tuple, form_eps[partner]))
        return jax.vmap(single)(Xb, Jb)
    return fn

def make_H_fn(bundle):
    k_arr = jnp.array({'e2':k_e2,'e5':k_e5,'neg':k_neg}[bundle], dtype=jnp.float32)
    bm = hym_models[bundle]
    @jax.jit
    def fn(Xb):
        def single(x):
            x_c = (x[:8] + 1j * x[8:]).astype(jnp.complex64)
            kappas = jnp.stack([jnp.sum(jnp.abs(x_c[2*i:2*i+2])**2) for i in range(4)])
            return (jnp.prod(kappas ** (-k_arr)).real) * jnp.exp(bm(x)[0]).real
        return jax.vmap(single)(Xb)
    return fn

CH = 256
nu_pb = {}; Hvals = {}
for fld, (partner, key, bundle, sector) in FIELD_MAP.items():
    Xj = jnp.array(CLOUDS[key]['X']); Jj = jnp.array(CLOUDS[key]['J'])
    k_tuple = tuple(int(v) for v in form_charges[partner])
    nfn = make_nu_pb_fn(partner, k_tuple); hfn = make_H_fn(bundle)
    nu = np.zeros((N_val,3),np.complex64); H = np.zeros(N_val,np.float64)
    for s in range(0, N_val, CH):
        e = min(s+CH, N_val)
        nu[s:e] = np.array(nfn(Xj[s:e], Jj[s:e])); H[s:e] = np.array(hfn(Xj[s:e]))
    nu_pb[fld] = nu; Hvals[fld] = H
    print(f"  {fld:3s} (<-{partner:2s}/{key:5s}/{bundle:3s}, sector {sector:6s}):  <|nu|>={np.abs(nu).mean():.4f}  H in [{H.min():.3f},{H.max():.3f}]")
print("All matter-field wavefunctions computed.")

# Save everything the normalisation notebook needs.
SECTORS = {                              # sector -> (field list, cloud key, family-block end or None)
    'H':     (['H'],            'id',   None),
    'Q':     (['Q1','Q2','Q3'], 'id',   2),
    'U':     (['U1','U2','U3'], 'id',   2),
    'E':     (['E1','E2','E3'], 'id',   2),     # reuses the up-sector form (e^c = u^c)
    'Hd':    (['Hd'],           '23',   None),
    'L_4_5': (['L1','L2'],      '1324', None),
    'D_4_5': (['D1','D2'],      '1324', None),
    'L_2_4': (['L3'],           '1324', None),
    'D_2_4': (['D3'],           '1324', None),
}
out = dict(
    nu_pb=nu_pb, Hvals=Hvals,
    clouds={k: dict(g=v['g'], ginv=v['ginv'], gref=v['gref']) for k,v in CLOUDS.items()},
    field_map=FIELD_MAP, sectors=SECTORS, factor_perm=FACTOR_PERM,
    aux=((y_val[:,0]/y_val[:,1])/6.0), V_CY=8.0,
    k_e2=k_e2, k_e5=k_e5, k_neg=k_neg,
)
with open('matter_fields_all.pkl','wb') as f:
    pickle.dump(out, f)
print("Saved matter_fields_all.pkl  (%d fields, %d clouds, N=%d points)" % (len(nu_pb), len(CLOUDS), N_val))
