#!/usr/bin/env python3
"""Hybrid feasibility score: polynomial probability + proximity features.

Features (all cheap, no oracle):
  logit(p12), d1, d5, dM, local feasible fraction within r=0.10,
  nearest-infeasible distance, ratio d_i/d_f.

Train a regularised logistic on the 32,768 survey labels (5-fold CV), compare
with poly3 and the raw proximity scores on the independent 32k scan, and check
the probability assigned to the measured top-20 feasible candidates.

  python -u tests/h1_hybrid_boundary.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import h5py, numpy as np
from scipy.spatial import cKDTree
BASE=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(BASE/'tests'))
from h1_surrogate_lib import BOUNDS, FREE, QC_MAX, load_library, normalised_currents, mahalanobis
import h1_boundary_model as bm
DATA=BASE/'tests'/'h1_full_scan'; A0,TH=0.10,0.12; BAND=0.02; LAM=0.02
OUT=DATA/'hybrid_boundary.json'

def feats(S,U,y,tf,ti,lib,exclude_self=False):
    # polynomial probability
    with h5py.File(DATA/'boundary_model.h5') as f: coef=f['coef'][:]; A=float(f.attrs['platt_A']); B=float(f.attrs['platt_B'])
    X,_=bm._poly(U,3,mu,sd); p12=bm._sigmoid(A*(X@coef)+B)
    tall=cKDTree(U)
    if not exclude_self:
        d1=tf.query(U,k=1)[0]; d5=tf.query(U,k=5)[0][:,-1]; di=ti.query(U,k=1)[0]
        nf=tf.query_ball_point(U,0.10,return_length=True); na=tall.query_ball_point(U,0.10,return_length=True)
    else:
        n=len(U); d1=np.empty(n); d5=np.empty(n); di=np.empty(n)
        nf=np.empty(n,int); na=np.empty(n,int)
        for i in range(n):
            if y[i]==1:
                q=tf.query(U[i],k=6)[0]; d1[i]=q[1]; d5[i]=q[5]; nf[i]=tf.query_ball_point(U[i],0.10,return_length=True)-1
            else:
                q=tf.query(U[i],k=5)[0]; d1[i]=q[0]; d5[i]=q[4]; nf[i]=tf.query_ball_point(U[i],0.10,return_length=True)
            if y[i]==0:
                di[i]=float(np.atleast_1d(ti.query(U[i],k=2)[0])[1])
            else:
                di[i]=float(np.atleast_1d(ti.query(U[i],k=1)[0])[0])
            na[i]=tall.query_ball_point(U[i],0.10,return_length=True)-1
    frac=np.asarray(nf,float)/(np.asarray(na,float)+1e-9)
    dm=np.array([mahalanobis(lib,S[i],A0) for i in range(len(S))])
    z=np.log(np.clip(p12,1e-9,1-1e-9)/(1-np.clip(p12,1e-9,1-1e-9)))
    F=np.column_stack([z,d1,d5,di,frac,dm,di/(d1+1e-9)])
    return F,p12

def fit(X,y,lam=LAM):
    def fg(w):
        z=X@w; q=1/(1+np.exp(-np.clip(z,-60,60)))
        f=np.mean(np.logaddexp(0,z)-y*z)+0.5*lam*w@w
        g=X.T@(q-y)/len(y)+lam*w
        return f,g
    from scipy.optimize import minimize
    return minimize(fg,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B').x

def pred(X,w): return 1/(1+np.exp(-np.clip(X@w,-60,60)))

def met(y,p,lc):
    band=((y==1)&(lc<TH+BAND))|((y==0)&(lc>=TH-BAND))
    o={'auc':float(bm.auc(y,p)),'ap':bm.average_precision(y,p),'band':float(bm.auc(y[band],p[band])) if band.sum()>10 else None}
    order=np.argsort(-p); tp=np.cumsum(y[order]); fp=np.cumsum(1-y[order]); tpr=tp/y.sum(); fpr=fp/(1-y).sum()
    for f in (0.05,0.10):
        k=int(np.searchsorted(fpr,f)); o[f'tpr5' if f==0.05 else 'tpr10']=float(tpr[min(k,len(tpr)-1)])
    return o

with h5py.File(DATA/'scan_all.h5') as f: Str=f['samples'][:]; ltr=f['scalars/lcfs_a'][:]
with h5py.File(DATA/'random_scan2.h5') as f: Sva=f['samples'][:]; lva=f['scalars/lcfs_a'][:]
Utr=normalised_currents(Str); Uva=normalised_currents(Sva)
ytr=(ltr>=TH).astype(int); yva=(lva>=TH).astype(int)
mu,sd=Utr.mean(0),Utr.std(0)+1e-12
lib=load_library(DATA/'surrogate_lib.h5')
tf=cKDTree(Utr[ytr==1]); ti=cKDTree(Utr[ytr==0])
Ftr,p12tr=feats(Str,Utr,ytr,tf,ti,lib,exclude_self=True); Fva,p12va=feats(Sva,Uva,yva,tf,ti,lib)
# standardise on train
mu2,sd2=Ftr.mean(0),Ftr.std(0)+1e-12; Ztr=(Ftr-mu2)/sd2; Zva=(Fva-mu2)/sd2
# 5-fold CV on survey
rng=np.random.default_rng(0); idx=rng.permutation(len(ytr)); folds=np.array_split(idx,5); cv=np.zeros(len(ytr))
for f in folds:
    tr=np.setdiff1d(idx,f); w=fit(Ztr[tr],ytr[tr]); cv[f]=pred(Ztr[f],w)
w=fit(Ztr,ytr)
Apl,Bpl=bm._platt(cv,ytr)
cv_cal=bm._sigmoid(Apl*bm._logit(cv)+Bpl)
te_raw=pred(Zva,w); te_cal=bm._sigmoid(Apl*bm._logit(te_raw)+Bpl)
res={'cv':met(ytr,cv,ltr),'cv_cal':met(ytr,cv_cal,ltr),
     'test':met(yva,te_cal,lva),
     'test_poly':met(yva,p12va,lva),'test_prox':met(yva,-Fva[:,1],lva)}
thr=bm._choose_thresholds(cv_cal,ytr)
print('CV hybrid ',res['cv']); print('CV calibrated',res['cv_cal']); print('test hybrid',res['test']); print('test poly ',res['test_poly']); print('test prox ',res['test_prox']); print('thresholds',thr)
# top-20 measured feasible candidates
cands=json.load(open(DATA/'near_optimal_top20_valid.json'))
Sc=np.array([c['currents_kA'] for c in cands])*1e3
Uc=normalised_currents(Sc); Fc,_=feats(Sc,Uc,np.ones(len(Sc)),tf,ti,lib); Zc=(Fc-mu2)/sd2; pc=pred(Zc,w)
print('\ntop candidates hybrid probability:')
for c,pp in zip(cands,pc): print(f"  idx {c['index']:6d} p_hybrid {pp:.4f} p12 {0:.4f} d1 {c['dm']:.2f}") if False else None
for c,pp in list(zip(cands,pc))[:5]:
    print(f"  idx {c['index']:6d} full {c['full_eps']:.4f} p_hybrid {pp:.4f}")
res['top_hybrid_mean']=float(np.mean(pc)); res['top_hybrid_min']=float(np.min(pc))
res['thresholds']=thr
json.dump(res,open(OUT,'w'),indent=2)
with h5py.File(DATA/'hybrid_boundary.h5','w') as f:
    f.create_dataset('w',data=w); f.create_dataset('mu',data=mu2); f.create_dataset('sd',data=sd2)
    f.attrs['platt_A']=Apl; f.attrs['platt_B']=Bpl
    f.attrs['thr_fpr5']=thr['fpr5']['threshold']; f.attrs['thr_fpr10']=thr['fpr10']['threshold']
    f.attrs['features']='logit_p12,d1,d5,di,frac_r0.1,dM,di/d1'
    f.attrs['a0']=A0
print('saved',OUT,'and hybrid_boundary.h5')
