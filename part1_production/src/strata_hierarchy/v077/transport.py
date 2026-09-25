"""STRATA v0.7.7 discrete directional transport.

The local norm does not by itself determine a unique connection. STRATA
therefore declares a discrete connection explicitly.

Because the Step-1 symmetric metric G is common to all Level-0 cells within a
specimen, tangent vectors are whitened by

    y = L^T v,   G = L L^T.

The local directional covector becomes

    q_x = L^{-1} b_x,

with ||q_x||_2 = ||b_x||_{G^-1}.

When q_i and q_j are both nonzero and not antipodal, transport from i to j is
the unique minimal plane rotation R_ij that maps

    q_i / ||q_i||  ->  q_j / ||q_j||

and is the identity on the orthogonal complement of their span.

Then

    T_ij v = L^{-T} R_ij L^T v.

T_ij is a G-isometry. It aligns preferred directional axes without inventing
a change in the symmetric norm. If directional strengths differ, beta is not
preserved; that mismatch is recorded explicitly.

Zero-zero edges use the symmetric identity transport.
One-sided and antipodal directional-frame cases are unresolved rather than
silently repaired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import eigh


ZERO_EPS=1e-12
ANTIPODAL_TOL=1e-10


def canonical_metric_sqrt(G):
    """Return the unique symmetric positive square root H and H^{-1}."""
    G=np.asarray(G,dtype=np.float64)
    Gs=0.5*(G+G.T)
    w,U=eigh(Gs,check_finite=True)
    if not np.isfinite(w).all() or np.min(w)<=0:
        raise ValueError("G must be finite symmetric positive definite")
    sw=np.sqrt(w)
    isw=1.0/sw
    H=(U*sw)@U.T
    Hinv=(U*isw)@U.T
    return H,Hinv,w


def whiten_covectors(G,B,chunk=2048):
    """Canonical symmetric whitening q = H^{-1} b, G = H^2."""
    G=np.asarray(G,dtype=np.float64)
    B=sparse.csr_matrix(B,dtype=np.float64)
    H,Hinv,w=canonical_metric_sqrt(G)

    Q=np.zeros((B.shape[0],B.shape[1]),dtype=np.float64)
    for a in range(0,B.shape[0],chunk):
        z=min(B.shape[0],a+chunk)
        X=B[a:z].toarray().T
        Q[a:z]=(Hinv@X).T

    rho=np.linalg.norm(Q,axis=1)
    return H,Hinv,Q,rho

def classify_pair(qi,qj,eps=ZERO_EPS,antipodal_tol=ANTIPODAL_TOL):
    ri=float(np.linalg.norm(qi))
    rj=float(np.linalg.norm(qj))

    if ri<=eps and rj<=eps:
        return {
            "transport_status":"IDENTITY_SYMMETRIC",
            "rho_i":ri,"rho_j":rj,
            "direction_cosine":np.nan,
            "direction_angle":np.nan,
            "directional_strength_mismatch":0.0,
        }

    if (ri<=eps) != (rj<=eps):
        return {
            "transport_status":"ONE_SIDED_UNRESOLVED",
            "rho_i":ri,"rho_j":rj,
            "direction_cosine":np.nan,
            "direction_angle":np.nan,
            "directional_strength_mismatch":abs(ri-rj),
        }

    a=qi/ri
    b=qj/rj
    c=float(np.clip(a@b,-1.0,1.0))
    angle=float(np.arccos(c))

    status=(
        "ANTIPODAL_UNRESOLVED"
        if c<=-1.0+antipodal_tol
        else "DIRECTIONAL_ROTATION"
    )
    return {
        "transport_status":status,
        "rho_i":ri,"rho_j":rj,
        "direction_cosine":c,
        "direction_angle":angle,
        "directional_strength_mismatch":abs(ri-rj),
    }


def rotate_minimal(a,b,x,antipodal_tol=ANTIPODAL_TOL):
    """Stable minimal Euclidean plane rotation mapping unit a to unit b.

    Rather than using the algebraically compact Rodrigues-like formula with
    division by (1+a.b), explicitly form the orthonormal 2-plane spanned by
    a and b. This avoids catastrophic amplification as a.b approaches -1.

    Antipodal pairs remain unresolved because the minimal rotation plane is
    not unique there.
    """
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    x=np.asarray(x,dtype=np.float64)

    na=float(np.linalg.norm(a))
    nb=float(np.linalg.norm(b))
    if na<=0 or nb<=0:
        raise ValueError("rotation directions must be nonzero")
    a=a/na
    b=b/nb

    c=float(np.clip(a@b,-1.0,1.0))

    if c>=1.0-1e-14:
        return x.copy()
    if c<=-1.0+antipodal_tol:
        raise ValueError("antipodal minimal rotation is not unique")

    # Build the second orthonormal basis direction in span(a,b).
    u=b-c*a
    sn=float(np.linalg.norm(u))
    if sn<=np.sqrt(np.finfo(np.float64).eps):
        # At this numerical scale the rotation plane cannot be resolved
        # reliably. This should only be reachable extremely near antipodal
        # or parallel cases, which are handled above.
        raise ValueError("directional rotation plane is numerically unresolved")
    e2=u/sn

    # Using the normalized plane makes c^2+s^2=1 to numerical precision.
    # Prefer sn as sin(theta); renormalize the pair to suppress dot-product
    # drift for near-antipodal directions.
    normcs=float(np.hypot(c,sn))
    c=c/normcs
    sn=sn/normcs

    x1=float(a@x)
    x2=float(e2@x)

    # Remove the original plane component and insert the rotated one:
    # a  -> c a + s e2 = b
    # e2 -> -s a + c e2
    x_perp=x-x1*a-x2*e2
    y1=c*x1-sn*x2
    y2=sn*x1+c*x2
    return x_perp+y1*a+y2*e2

def apply_transport(H,Hinv,qi,qj,v):
    info=classify_pair(qi,qj)
    status=info["transport_status"]

    if status=="IDENTITY_SYMMETRIC":
        return np.asarray(v,dtype=np.float64).copy()
    if status!="DIRECTIONAL_ROTATION":
        raise ValueError(f"transport unresolved: {status}")

    ri=info["rho_i"]; rj=info["rho_j"]
    a=qi/ri; b=qj/rj

    y=H@np.asarray(v,dtype=np.float64)
    yp=rotate_minimal(a,b,y)
    return Hinv@yp

def transport_audit_one(G,H,Hinv,qi,qj,seed=0):
    info=classify_pair(qi,qj)
    status=info["transport_status"]

    if status=="IDENTITY_SYMMETRIC":
        return {
            **info,
            "alpha_isometry_error":0.0,
            "direction_map_error":0.0,
            "roundtrip_error":0.0,
            "certificate_pass":True,
        }

    if status!="DIRECTIONAL_ROTATION":
        return {
            **info,
            "alpha_isometry_error":np.nan,
            "direction_map_error":np.nan,
            "roundtrip_error":np.nan,
            "certificate_pass":True,  # correctly unresolved
        }

    rng=np.random.default_rng(seed)
    v=rng.normal(size=G.shape[0])

    vt=apply_transport(H,Hinv,qi,qj,v)
    vback=apply_transport(H,Hinv,qj,qi,vt)

    a0=float(np.sqrt(v@(G@v)))
    a1=float(np.sqrt(vt@(G@vt)))
    alpha_err=float(abs(a1-a0)/(max(a0,1e-15)))

    # Check preferred-axis map in whitened coordinates.
    ai=qi/np.linalg.norm(qi)
    aj=qj/np.linalg.norm(qj)
    mapped=rotate_minimal(ai,aj,ai)
    dir_err=float(np.linalg.norm(mapped-aj))

    rt=float(np.linalg.norm(vback-v)/(max(np.linalg.norm(v),1e-15)))

    return {
        **info,
        "alpha_isometry_error":alpha_err,
        "direction_map_error":dir_err,
        "roundtrip_error":rt,
        "certificate_pass":bool(
            alpha_err<=1e-10
            and dir_err<=1e-10
            and rt<=1e-10
        ),
    }


def build_transport_registry(edges,Q,rho):
    i=edges.super_i.to_numpy(np.int64)
    j=edges.super_j.to_numpy(np.int64)

    rows=[]
    for k,(u,v) in enumerate(zip(i,j)):
        info=classify_pair(Q[u],Q[v])
        rows.append({
            "super_i":int(u),
            "super_j":int(v),
            **info,
        })
    return pd.DataFrame(rows)
