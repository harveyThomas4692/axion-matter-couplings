#!/usr/bin/env python3
# Ported from model2_train.ipynb (repository root).
# The notebook cell code is VERBATIM except for lines tagged [SCAN] (the scan harness).
# Stage 1 (model 2): generate-or-load the per-t3 point cloud at the D-flat Kahler point,
#          train the metric + 2 HYM + 8 sigma nets, save model2_run_<ts>.pkl into --out-dir.
import argparse, os
import matplotlib
matplotlib.use('Agg')

_ap = argparse.ArgumentParser()
_ap.add_argument("--t3", type=float, required=True,
                 help="[SCAN] D-flat shape modulus; t2=1, t1=(t3-2)(t3+1), t4=t1/t3. Needs t3>2.")
_ap.add_argument("--seed", type=int, required=True)
_ap.add_argument("--n-points", dest="n_points", type=int, default=500000)
_ap.add_argument("--epochs", type=int, default=25, help="per-net training epochs (notebook default 25)")
_ap.add_argument("--cloud-dir", dest="cloud_dir", required=True,
                 help="per-t3 cloud dir (generated if basis.pickle absent)")
_ap.add_argument("--out-dir", dest="out_dir", required=True,
                 help="per-(t3,seed) run dir for the saved pkl")
_ap.add_argument("--parity", choices=["corrected", "legacy"], default="corrected",
                 help="[SCAN] 'corrected' (default) is the notebooks' CURRENT table, taken "
                      "verbatim: eps=(-1)^(a+b+1) for L5, (-1)^(b+1) for fb. 'legacy' restores "
                      "the retired pre-2026-08-13 table whose sigma projection cannot fit, and "
                      "exists only to reproduce the parity A/B experiment. The old spelling "
                      "'notebook' was removed: after 2026-08-13 it would mean its own opposite.")
ARGS = _ap.parse_args()
if ARGS.t3 <= 2.0:
    raise SystemExit(f"[SCAN] t3 must exceed 2 for t1>0 (Kahler cone); got {ARGS.t3}")
os.makedirs(ARGS.out_dir, exist_ok=True)
import numpy as _np
_np.random.seed(ARGS.seed)  # [SCAN] reproducible np permutations in training

import numpy as np
import os, pickle, itertools, datetime
import jax, jax.numpy as jnp, jax.nn as jnn
import equinox as eqx, optax
from tqdm import tqdm
import matplotlib.pyplot as plt
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator  # [SCAN] numpy generator (no Mathematica)
from cymetric.jax.models.models import PhiFSModel
from cymetric.jax.models.helper import train_model
from cymetric.jax.models.helper import prepare_basis as jax_prepare_basis
from cymetric.jax.models.callbacks import SigmaCallback

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

# --- Z2 (model 2): a single generator, the projective flip _g1.  OUTPUT-averaged. ---
_G1IDX = jnp.array([1, 3, 5, 7, 9, 11, 13, 15])
def _g1(z): return z.at[_G1IDX].multiply(-1.0)
class OrbitAvg(eqx.Module):
    net: eqx.Module
    def __call__(self, x):
        return 0.5 * ( self.net(x) + self.net(_g1(x)) )        # Z2, not Z2xZ2
print("Z2 orbit-average + features defined.")

key = jax.random.PRNGKey(ARGS.seed)  # [SCAN] error-bar seed
psi0, psi1 = 2.0, 1.0                      # same complex structure as the tetra-quadric
ambient = np.array([1, 1, 1, 1])
alpha   = [1., 0., 1., 0., 0.]
combinations = list(itertools.product([0, 1], repeat=4))

# ---- D-flat Kahler point: specify t3, solve t1,t4, check the cone ----
t2 = 1.0
t3 = ARGS.t3  # [SCAN] scan variable: the D-flat shape modulus (needs t3>2)
t1 = (t3 - 2.0)*(t3 + 1.0)                  # mu(L4)=0 at t2=1
t4 = t1 / t3                                # mu(L1)=0 : t1 t2 = t3 t4
kmoduli = np.array([t1, t2, t3, t4], dtype=float)
assert np.all(kmoduli > 0), f"NOT in Kahler cone: {kmoduli}"
print("kmoduli (D-flat point) =", kmoduli)

# ---- matter bundles (the two the axion couples to) ----
K_L5 = np.array([2., 1., 0., -2.])         # the 10s
K_FB = np.array([0., 1., -2., 0.])         # the 5-bar
hym_names   = ['L5', 'fb']
hym_charges = [K_L5, K_FB]

n_cy_pts = ARGS.n_points  # [SCAN]
dirname  = ARGS.cloud_dir  # [SCAN] per-t3 cloud (kmoduli varies along the scan)

# Defining polynomial (identical to the tetra-quadric CS)
_mons, _cfs = [], []
for c in combinations:
    _mons.append([2*(1-c[0]),2*c[0], 2*(1-c[1]),2*c[1], 2*(1-c[2]),2*c[2], 2*(1-c[3]),2*c[3]])
    _cfs.append(1. if sum(c)%2==0 else psi0)
_mons.append([1,1,1,1,1,1,1,1]); _cfs.append(psi1)
monomials    = np.array(_mons, dtype=np.int64)
coefficients = np.array(_cfs)

pg = CICYPointGenerator([monomials],[coefficients], kmoduli, ambient)  # [SCAN]
basis_path = os.path.join(dirname, 'basis.pickle')
if not os.path.exists(basis_path):
    print(f"Generating {n_cy_pts} points at the D-flat kmoduli ...")
    kappa = pg.prepare_dataset(n_cy_pts, dirname); pg.prepare_basis(dirname, kappa)
with open(basis_path,'rb') as f: BASIS_raw = pickle.load(f)
BASIS = jax_prepare_basis(BASIS_raw)
data = np.load(os.path.join(dirname,'dataset.npz'))
X_train,y_train,X_val,y_val = data['X_train'],data['y_train'],data['X_val'],data['y_val']
n_train = X_train.shape[0]
print("X_train",X_train.shape,"X_val",X_val.shape)

keys = jax.random.split(key, 8); key = keys[0]
metric_nn = eqx.nn.Sequential([
    eqx.nn.Lambda(kappa_features),
    eqx.nn.Linear(16,128,key=keys[1]), eqx.nn.Lambda(jnn.gelu),
    eqx.nn.Linear(128,128,key=keys[2]), eqx.nn.Lambda(jnn.gelu),
    eqx.nn.Linear(128,128,key=keys[3]), eqx.nn.Lambda(jnn.gelu),
    eqx.nn.Linear(128,1,use_bias=False,key=keys[4]),
])
metric_model = PhiFSModel(OrbitAvg(metric_nn), BASIS, alpha=alpha)
metric_model = eqx.tree_at(lambda m: m.learn_volk,       metric_model, replace=False)
metric_model = eqx.tree_at(lambda m: m.learn_transition, metric_model, replace=False)
metric_model, hist = train_model(metric_model, data, optimizer=optax.adam(1e-3),
                                 epochs=ARGS.epochs, callbacks=[SigmaCallback((X_val,y_val))],  # [SCAN]
                                 batch_sizes=(128,256))

@jax.jit
def compute_chunk(Xc):
    g = metric_model(Xc); J = metric_model.pullbacks(Xc); gi = jnp.linalg.inv(g)
    xc = (Xc[:,:8]+1j*Xc[:,8:]).astype(jnp.complex64)
    return gi, J, xc
def gref_tr(gi, J, xc, i):
    Ji=J[:,:,2*i:2*i+2]; xi=xc[:,2*i:2*i+2]; ki=jnp.sum(jnp.abs(xi)**2,axis=1,keepdims=True)
    JiJi=jnp.einsum('...ai,...bi->...ab',Ji,jnp.conj(Ji)); Jixi=jnp.einsum('...ai,...i->...a',Ji,xi)
    out=jnp.einsum('...a,...b->...ab',Jixi,jnp.conj(Jixi))
    gref=(ki[:,:,None]*JiJi - out)/(ki**2)[:,:,None]
    return jnp.real(jnp.trace(gi@gref,axis1=-2,axis2=-1))
CHUNK=512
hym_rho={n:np.zeros(n_train,np.float32) for n in hym_names}
J_all=np.zeros((n_train,3,8),np.complex64); ginv_all=np.zeros((n_train,3,3),np.complex64)
for s in tqdm(range(0,n_train,CHUNK)):
    e=min(s+CHUNK,n_train); Xc=jnp.array(X_train[s:e],jnp.float32)
    gi,J,xc=compute_chunk(Xc); tr=[gref_tr(gi,J,xc,i) for i in range(4)]
    for nm,kv in zip(hym_names,hym_charges):
        hym_rho[nm][s:e]=sum(float(kv[i])*np.array(tr[i]) for i in range(4)).astype(np.float32)
    J_all[s:e]=np.array(J); ginv_all[s:e]=np.array(gi)
print("HYM sources ready:", {n:round(float(hym_rho[n].mean()),4) for n in hym_names})

def M_to_omega(M):
    R=jnp.real(M)/4.; I=jnp.imag(M)/4.
    return jnp.concatenate([jnp.concatenate([R,-I],1), jnp.concatenate([I,R],1)],0)
def cy_laplacian(f, x, Om):
    gfn=jax.grad(f)
    HO=jax.vmap(lambda c: jax.jvp(gfn,(x,),(c,))[1], in_axes=1, out_axes=0)(Om)
    return jnp.trace(HO)
def make_hym_nn(sk):
    ks=jax.random.split(sk,5)
    return eqx.nn.Sequential([eqx.nn.Lambda(kappa_features),
        eqx.nn.Linear(16,128,key=ks[1]),eqx.nn.Lambda(jnn.gelu),
        eqx.nn.Linear(128,128,key=ks[2]),eqx.nn.Lambda(jnn.gelu),
        eqx.nn.Linear(128,128,key=ks[3]),eqx.nn.Lambda(jnn.gelu),
        eqx.nn.Linear(128,1,use_bias=False,key=ks[4])])
@eqx.filter_jit
def hym_step(model,Xb,Jb,gib,rhob,wb,opt_state,optimizer):
    def loss_fn(m_):
        def pp(x,Ji,gii,rho):
            M=jnp.conj(Ji).T@gii@Ji; lap=cy_laplacian(lambda z:m_(z)[0],x,M_to_omega(M))
            return jnp.abs(lap-rho)
        pw=jax.vmap(pp)(Xb,Jb,gib,rhob); return jnp.sum(wb*pw)/jnp.sum(wb)
    loss,gr=eqx.filter_value_and_grad(loss_fn)(model)
    upd,os_=optimizer.update(gr,opt_state,eqx.filter(model,eqx.is_array))
    return eqx.apply_updates(model,upd),os_,loss
w_all=y_train[:,0].astype(np.float32)
hym_models={}; hym_histories={}
for bi,nm in enumerate(hym_names):
    key,sk=jax.random.split(key); m=OrbitAvg(make_hym_nn(sk))
    opt=optax.adam(1e-3); os_=opt.init(eqx.filter(m,eqx.is_array))
    ref=float(np.sum(w_all*np.abs(hym_rho[nm]))/np.sum(w_all)); h=[]
    print(f"HYM {nm}:")
    for ep in range(ARGS.epochs):  # [SCAN]
        perm=np.random.permutation(n_train); el=0.; nb=0
        for s in range(0,n_train,128):
            idx=perm[s:s+128]
            m,os_,loss=hym_step(m, jnp.array(X_train[idx],jnp.float32), jnp.array(J_all[idx],jnp.complex64),
                                jnp.array(ginv_all[idx],jnp.complex64), jnp.array(hym_rho[nm][idx],jnp.float32),
                                jnp.array(w_all[idx],jnp.float32), os_, opt)
            el+=float(loss); nb+=1
        h.append(el/nb/ref); print(f"  ep{ep+1:2d}: M_HYM={h[-1]:.4f}")
    hym_models[nm]=m; hym_histories[nm]=h

def _mu_bar(x_cplx, j):
    v=jnp.zeros(8,jnp.complex64)
    v=v.at[2*j].set(-jnp.conj(x_cplx[2*j+1])); v=v.at[2*j+1].set(jnp.conj(x_cplx[2*j])); return v

# field -> (bundle, form params, eps).  eps = the form's g1-eigenvalue: (-1)^(a+b+1) for L5, (-1)^(b+1) for fb.
# (Wilson W=diag(1,1,1,-1,-1), rho_a=+1 then label each form as an MSSM field.)
FORM_SPEC = {
    'U1':('L5',(0,0),-1),'U2':('L5',(1,1),-1),'U3':('L5',(2,0),-1),     # u^c (=e^c)
    'Q1':('L5',(0,1),+1),'Q2':('L5',(1,0),+1),'Q3':('L5',(2,1),+1),     # Q
    'D' :('fb',(0,), -1),                                               # d^c
    'L' :('fb',(1,), +1),                                               # lepton doublet
}
_FORM_SPEC_LEGACY = {  # [SCAN] the RETIRED pre-2026-08-13 table; --parity legacy only
    'U1':('L5',(0,0),+1),'U2':('L5',(1,1),+1),'U3':('L5',(2,0),+1),  # [SCAN] u^c (=e^c) : parity +1
    'Q1':('L5',(0,1),-1),'Q2':('L5',(1,0),-1),'Q3':('L5',(2,1),-1),  # [SCAN] Q : parity -1
    'D' :('fb',(0,), +1),  # [SCAN] d^c : parity +1
    'L' :('fb',(1,), -1),  # [SCAN] lepton : parity -1
}
if ARGS.parity == "legacy":  # [SCAN] reproduce the buggy pre-fix parity, for the A/B only
    FORM_SPEC = _FORM_SPEC_LEGACY  # [SCAN]
form_names   = list(FORM_SPEC)
form_bundle  = {f:FORM_SPEC[f][0] for f in form_names}
form_eps     = {f:FORM_SPEC[f][2] for f in form_names}
form_charges = {f:(K_L5 if FORM_SPEC[f][0]=='L5' else K_FB) for f in form_names}

def nu_ref(x_cplx, form_name):
    bnd, prm, _ = FORM_SPEC[form_name]
    if bnd=='L5':
        a,b=prm; kap=jnp.sum(jnp.abs(x_cplx[6:8])**2)
        poly=x_cplx[0]**(2-a)*x_cplx[1]**a * x_cplx[2]**(1-b)*x_cplx[3]**b
        return (poly/kap**2)*_mu_bar(x_cplx,3), K_L5
    else:
        b=prm[0]; kap=jnp.sum(jnp.abs(x_cplx[4:6])**2)
        poly=x_cplx[2]**(1-b)*x_cplx[3]**b
        return (poly/kap**2)*_mu_bar(x_cplx,2), K_FB

def log_H_grad(x_cplx, k_vec):
    g=jnp.zeros(8,jnp.complex64)
    for i in range(4):
        xi=x_cplx[2*i:2*i+2]; kap=jnp.sum(jnp.abs(xi)**2)
        g=g.at[2*i:2*i+2].set(-k_vec[i]*jnp.conj(xi)/kap)
    return g
print("Reference forms (type-I) defined:", form_names)

# Pre-compute sigma sources rho_nu (HYM bundle metric H = e^beta H_ref)
def make_HNu(fn):
    def f(xr):
        xc=(xr[:8]+1j*xr[8:]).astype(jnp.complex64); nu,kv=nu_ref(xc,fn)
        kap=jnp.array([jnp.sum(jnp.abs(xc[2*i:2*i+2])**2) for i in range(4)])
        H=jnp.prod(kap**(-jnp.array(kv)))*jnp.exp(hym_models[form_bundle[fn]](xr)[0])
        HNu=H*nu; return jnp.concatenate([jnp.real(HNu),jnp.imag(HNu)])
    return f
def make_H(fn):
    def f(xc):
        _,kv=nu_ref(xc,fn); kap=jnp.array([jnp.sum(jnp.abs(xc[2*i:2*i+2])**2) for i in range(4)])
        xr=jnp.concatenate([jnp.real(xc),jnp.imag(xc)])
        return jnp.prod(kap**(-jnp.array(kv)))*jnp.exp(hym_models[form_bundle[fn]](xr)[0])
    return f
def make_source(fn):
    HNu=make_HNu(fn); H=make_H(fn)
    @jax.jit
    def sb(Xc,gic,Jc):
        xc=(Xc[:,:8]+1j*Xc[:,8:]).astype(jnp.complex64)
        d=jax.vmap(jax.jacfwd(HNu))(Xc)
        dw=(0.5*(d[:,:8,:8]+d[:,8:,8:])+0.5j*(d[:,8:,:8]-d[:,:8,8:])).astype(jnp.complex64)
        T=jnp.einsum('xAb,xab,xBa->xAB', Jc, dw, jnp.conj(Jc))
        Hv=jax.vmap(H)(xc)
        return -jnp.trace(gic@T,axis1=-2,axis2=-1)/Hv
    return sb
source_fns={f:make_source(f) for f in form_names}
sigma_rho={f:np.zeros(n_train,np.complex64) for f in form_names}
print("Pre-computing sigma sources ...")
for s in tqdm(range(0,n_train,512)):
    e=min(s+512,n_train); Xc=jnp.array(X_train[s:e],jnp.float32)
    gic=jnp.array(ginv_all[s:e],jnp.complex64); Jc=jnp.array(J_all[s:e],jnp.complex64)
    for f in form_names: sigma_rho[f][s:e]=np.array(source_fns[f](Xc,gic,Jc))
print("sigma sources ready.")

# Train the 8 harmonic-form (sigma) nets, projected onto the Z2 parity eps of each form
def n_sections(k):
    n=1
    for ki in k: n*=abs(ki)+1
    return n
def build_sections(xc,k):
    per=[]
    for i,ki in enumerate(k):
        xi=xc[2*i:2*i+2]; m=abs(int(ki))
        if ki>=0: segs=[xi[0]**(m-a)*xi[1]**a for a in range(m+1)]
        else:
            kap=jnp.sum(jnp.abs(xi)**2); segs=[jnp.conj(xi[0])**(m-a)*jnp.conj(xi[1])**a/kap**m for a in range(m+1)]
        per.append(jnp.stack(segs))
    r=per[0]
    for s in per[1:]: r=jnp.outer(r,s).reshape(-1)
    return r.astype(jnp.complex64)
def make_sigma_nn(sk,nout):
    ks=jax.random.split(sk,5)
    return eqx.nn.Sequential([eqx.nn.Lambda(kappa_features),
        eqx.nn.Linear(16,128,key=ks[1]),eqx.nn.Lambda(jnn.gelu),
        eqx.nn.Linear(128,128,key=ks[2]),eqx.nn.Lambda(jnn.gelu),
        eqx.nn.Linear(128,128,key=ks[3]),eqx.nn.Lambda(jnn.gelu),
        eqx.nn.Linear(128,nout,use_bias=False,key=ks[4])])
def make_sigma_step(k_tuple, eps, beta_model):
    N=n_sections(k_tuple); kj=jnp.array(k_tuple,jnp.float32); e1=jnp.float32(eps)
    def _raw(m_,x):
        xc=(x[:8]+1j*x[8:]).astype(jnp.complex64); s=build_sections(xc,k_tuple)
        out=m_(x); c=out[:N]+1j*out[N:]; return jnp.dot(c,s)
    def sig(m_,x): return 0.5*(_raw(m_,x)+e1*_raw(m_,_g1(x)))       # Z2 parity projection
    @eqx.filter_jit
    def step(model,Xb,Jb,gib,rhob,wb,opt_state,optimizer):
        def loss_fn(m_):
            def pp(x,Ji,gii,rho):
                xc=(x[:8]+1j*x[8:]).astype(jnp.complex64); M=jnp.conj(Ji).T@gii@Ji; Om=M_to_omega(M)
                dlnH=log_H_grad(xc,kj)
                gb=jax.grad(lambda z:beta_model(z)[0])(x); dlnH=dlnH+0.5*(gb[:8]-1j*gb[8:])
                v=jnp.conj(Ji).T@(gii@(Ji@dlnH))
                def sR(z): return jnp.real(sig(m_,z))
                def sI(z): return jnp.imag(sig(m_,z))
                gR=jax.grad(sR)(x); gI=jax.grad(sI)(x)
                db=0.5*((gR[:8]+1j*gR[8:])+1j*(gI[:8]+1j*gI[8:]))
                lap=cy_laplacian(sR,x,Om)+1j*cy_laplacian(sI,x,Om)
                return jnp.abs(lap+jnp.sum(v*db)-rho)
            pw=jax.vmap(pp)(Xb,Jb,gib,rhob); return jnp.sum(wb*pw)/jnp.sum(wb)
        loss,gr=eqx.filter_value_and_grad(loss_fn)(model)
        upd,os_=optimizer.update(gr,opt_state,eqx.filter(model,eqx.is_array))
        return eqx.apply_updates(model,upd),os_,loss
    return step

sigma_models={}; sigma_histories={}
for f in form_names:
    k_tuple=tuple(int(v) for v in form_charges[f]); N=n_sections(k_tuple)
    key,sk=jax.random.split(key); m=make_sigma_nn(sk,2*N)
    step=make_sigma_step(k_tuple, form_eps[f], hym_models[form_bundle[f]])
    opt=optax.adam(1e-3); os_=opt.init(eqx.filter(m,eqx.is_array))
    ref=float(np.sum(w_all*np.abs(sigma_rho[f]))/np.sum(w_all)); h=[]
    print(f"sigma {f} (bundle {form_bundle[f]}, eps={form_eps[f]:+d}):")
    for ep in range(ARGS.epochs):  # [SCAN]
        perm=np.random.permutation(n_train); el=0.; nb=0
        for s in range(0,n_train,128):
            idx=perm[s:s+128]
            m,os_,loss=step(m, jnp.array(X_train[idx],jnp.float32), jnp.array(J_all[idx],jnp.complex64),
                            jnp.array(ginv_all[idx],jnp.complex64), jnp.array(sigma_rho[f][idx],jnp.complex64),
                            jnp.array(w_all[idx],jnp.float32), os_, opt)
            el+=float(loss); nb+=1
        h.append(el/nb/ref); print(f"  ep{ep+1:2d}: M_sigma={h[-1]:.4f}")
    sigma_models[f]=m; sigma_histories[f]=h

timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
fn=os.path.join(ARGS.out_dir, f'model2_run_{timestamp}.pkl')  # [SCAN] per-run dir
def to_np(m): return jax.tree_util.tree_map(lambda x: np.asarray(x) if hasattr(x,'shape') else x, m)
_full=np.load(os.path.join(dirname,'dataset.npz'))
val_pb=_full['val_pullbacks'] if 'val_pullbacks' in _full else None
save={
  'metadata':{'timestamp':timestamp,'psi0':float(psi0),'psi1':float(psi1),
              'kmoduli':kmoduli.tolist(),'ambient':ambient.tolist(),'dirname':dirname,
              'n_train':int(n_train),'K_L5':K_L5.tolist(),'K_FB':K_FB.tolist(),
              'hym_names':hym_names,'form_names':form_names,
              'form_charges':{k:v.tolist() for k,v in form_charges.items()},
              'form_bundle':form_bundle,'form_eps':form_eps,
              'monomials':monomials.tolist(),'coefficients':coefficients.tolist()},
  'models':{'metric_model':to_np(metric_model),
            'hym_models':{k:to_np(v) for k,v in hym_models.items()},
            'sigma_models':{k:to_np(v) for k,v in sigma_models.items()}},
  'training_data':{'X_train':np.asarray(X_train),'y_train':np.asarray(y_train),
                   'X_val':np.asarray(X_val),'y_val':np.asarray(y_val),'val_pullbacks':val_pb},
  'precomputed':{'J_all':J_all,'ginv_all':ginv_all,
                 'hym_rho':{k:v for k,v in hym_rho.items()},
                 'sigma_rho':{k:v for k,v in sigma_rho.items()}},
  'histories':{'hym_histories':hym_histories,'sigma_histories':sigma_histories},
  'BASIS_raw':BASIS_raw,
}
with open(fn,'wb') as f:
    cd=dict(save); cd['models']=dict(save['models'])
    cd['models']['metric_model']=eqx.tree_at(lambda m:m._sigma_loss_fn, cd['models']['metric_model'], replace=[])
    pickle.dump(cd, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved", fn, "(%.1f MB)"%(os.path.getsize(fn)/1024**2))
print("  metric @ kmoduli =", kmoduli, " | HYM:", list(hym_models), " | sigma:", list(sigma_models))
