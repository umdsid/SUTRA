from __future__ import annotations
from dataclasses import dataclass
import hashlib, json
import numpy as np
from scipy import sparse

@dataclass(frozen=True)
class SymmetricBaseConfig:
    mechanics_tension_weight: float = 1.0
    mechanics_delta_p_weight: float = 1.0
    eig_floor: float = 1e-10
    max_condition_number: float = 1e8

def canonical_feature_order(genes):
    return tuple(str(g) for g in genes)+("__tension_z__","__delta_pressure_z__")

def feature_order_sha256(features):
    payload=json.dumps(list(features),ensure_ascii=False,separators=(",",":")).encode()
    return hashlib.sha256(payload).hexdigest()

def build_symmetric_base_metric(functional_laplacian,lambda_g,n_genes,cfg=SymmetricBaseConfig()):
    if n_genes<=0: raise ValueError("n_genes must be positive")
    if lambda_g<0: raise ValueError("lambda_g must be nonnegative")
    if cfg.mechanics_tension_weight<=0 or cfg.mechanics_delta_p_weight<=0:
        raise ValueError("mechanics weights must be positive")
    L=sparse.csr_matrix(functional_laplacian,dtype=np.float64)
    if L.shape!=(n_genes,n_genes):
        raise ValueError(f"functional Laplacian shape {L.shape} != {(n_genes,n_genes)}")
    Gmol=sparse.eye(n_genes,format="csr")+lambda_g*L
    mech=sparse.diags(
        [cfg.mechanics_tension_weight,cfg.mechanics_delta_p_weight],
        offsets=0,format="csr",dtype=np.float64
    )
    return sparse.block_diag((Gmol,mech),format="csr")

def alpha_norm(G,v):
    v=np.asarray(v,dtype=np.float64)
    if G.shape[0]!=len(v): raise ValueError("metric/vector dimension mismatch")
    q=float(v@(G@v))
    if q<-1e-10: raise ValueError(f"negative quadratic form: {q}")
    return float(np.sqrt(max(0.0,q)))

def certify_symmetric_base_metric(G,cfg=SymmetricBaseConfig()):
    A=sparse.csr_matrix(G,dtype=np.float64)
    dense=A.toarray()
    asym=float(np.max(np.abs(dense-dense.T)))
    sym=0.5*(dense+dense.T)
    eig=np.linalg.eigvalsh(sym)
    lmin=float(eig[0]); lmax=float(eig[-1])
    cond=float(lmax/lmin) if lmin>0 else float("inf")
    finite=bool(np.isfinite(eig).all() and np.isfinite(cond))
    positive=bool(lmin>cfg.eig_floor)
    conditioned=bool(cond<=cfg.max_condition_number)
    return {
        "dimension":int(A.shape[0]),
        "symmetry_max_abs_error":asym,
        "lambda_min":lmin,
        "lambda_max":lmax,
        "condition_number":cond,
        "finite":finite,
        "positive_definite":positive,
        "conditioned":conditioned,
        "certificate_pass":bool(finite and positive and conditioned and asym<=1e-10),
    }

def sparse_matrix_sha256(G):
    A=sparse.csr_matrix(G,dtype=np.float64)
    h=hashlib.sha256()
    for arr in (
        np.asarray(A.shape,dtype=np.int64),
        A.indptr.astype(np.int64,copy=False),
        A.indices.astype(np.int64,copy=False),
        A.data.astype(np.float64,copy=False),
    ):
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()
