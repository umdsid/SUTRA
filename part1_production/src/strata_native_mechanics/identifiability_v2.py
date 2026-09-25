from dataclasses import dataclass
from collections import Counter
import numpy as np
from scipy.sparse import coo_matrix, hstack
from scipy.sparse.csgraph import structural_rank

@dataclass(frozen=True)
class BoundaryReductionConfig:
    small_gap_pixels: int = 64
    medium_gap_pixels: int = 256

def rank_nullity(A):
    r = int(structural_rank(A)); n = int(A.shape[1])
    return {"n_rows": int(A.shape[0]), "n_variables": n, "structural_rank": r,
            "structural_nullity": max(n-r, 0),
            "structural_rank_fraction": r/max(n,1),
            "structurally_identifiable": r >= n}

def nullity_attribution(A, meta):
    base = rank_nullity(A); n = A.shape[1]
    classes = {
        "tension": np.array(sorted(meta["eidx"].values()), int),
        "cell_pressure": np.array(sorted(meta["cidx"].values()), int),
        "boundary_pressure": np.array(sorted(meta["bidx"].values()), int),
    }
    out = {}
    for name, idx in classes.items():
        keep = np.ones(n, bool); keep[idx] = False
        q = rank_nullity(A[:, keep])
        out[name] = {
            "n_variables_in_class": int(len(idx)),
            "nullity_without_class": int(q["structural_nullity"]),
            "nullity_removed_if_class_removed":
                int(base["structural_nullity"] - q["structural_nullity"]),
        }
    return base, out

def equation_counts(meta):
    c = Counter(str(x[0]) for x in meta.get("rows", []))
    return {
        "young_laplace_rows": int(c.get("young_laplace", 0)),
        "junction_x_rows": int(c.get("junction_x", 0)),
        "junction_y_rows": int(c.get("junction_y", 0)),
    }

def _groups(meta, bg, mode, cfg):
    info = {int(r.background_component):(str(r.kind), int(r.pixels))
            for r in bg.itertuples()} if len(bg) else {}
    out = {}; nxt = 0; ext = small = medium = None
    for bc in sorted(int(x) for x in meta["bidx"]):
        kind, pix = info.get(bc, ("internal_gap", 10**9))
        if mode == "independent":
            out[bc] = nxt; nxt += 1; continue
        if kind == "exterior":
            if ext is None: ext = nxt; nxt += 1
            out[bc] = ext; continue
        if mode == "exterior_shared":
            out[bc] = nxt; nxt += 1
        elif mode == "small_gap_shared":
            if pix <= cfg.small_gap_pixels:
                if small is None: small = nxt; nxt += 1
                out[bc] = small
            else:
                out[bc] = nxt; nxt += 1
        elif mode == "tiered_gap_shared":
            if pix <= cfg.small_gap_pixels:
                if small is None: small = nxt; nxt += 1
                out[bc] = small
            elif pix <= cfg.medium_gap_pixels:
                if medium is None: medium = nxt; nxt += 1
                out[bc] = medium
            else:
                out[bc] = nxt; nxt += 1
        else:
            raise ValueError(mode)
    return out

def reduce_boundary(A, meta, bg, mode, cfg):
    npre = len(meta["eidx"]) + len(meta["cidx"])
    gm = _groups(meta, bg, mode, cfg)
    if not gm:
        return A.copy(), {"mode": mode, "n_boundary_variables_original": 0,
                          "n_boundary_variables_reduced": 0}
    old = sorted(meta["bidx"].items(), key=lambda kv: kv[1])
    nG = max(gm.values()) + 1
    rr=[]; cc=[]; vv=[]
    for bc, acol in old:
        rr.append(acol - npre); cc.append(gm[int(bc)]); vv.append(1.0)
    G = coo_matrix((vv,(rr,cc)), shape=(len(old), nG)).tocsr()
    R = hstack([A[:,:npre], A[:,npre:] @ G], format="csr")
    return R, {"mode": mode, "n_boundary_variables_original": len(old),
               "n_boundary_variables_reduced": nG}

def reduction_audit(A, meta, bg, cfg):
    base = rank_nullity(A); rows=[]
    for mode in ("independent","exterior_shared","small_gap_shared","tiered_gap_shared"):
        R,m = reduce_boundary(A,meta,bg,mode,cfg)
        q = rank_nullity(R)
        rows.append({**m, **q,
                     "nullity_reduction_vs_independent":
                         int(base["structural_nullity"] - q["structural_nullity"])})
    return rows
