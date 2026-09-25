"""Dependency-aware active-block preflight for STRATA v0.7.1.

Implements the Level-0 preparation required before recursive hierarchy
construction:

- full measured gene state is retained;
- GO/MSigDB-style GMT resources may supply the functional prior;
- mechanics remains primitive tension/delta-p with explicit missingness;
- signaling-resource schema is audited separately from functional resources;
- robust Level-0 scales are frozen and are not hierarchy-level adaptive.

No hierarchy aggregation is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import re

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


SIGNALING_NAME_TOKENS = (
    "cellchat", "omnipath", "ligand", "receptor", "signaling", "signal",
)

FUNCTIONAL_NAME_TOKENS = (
    "msig", "hallmark", "reactome", "go.", "go_", "c5", "c2", "h.all",
)

SENDER_ALIASES = (
    "sender_gene", "ligand", "ligand_gene", "source_gene", "from_gene",
)
RECEIVER_ALIASES = (
    "receiver_gene", "receptor", "receptor_gene", "target_gene", "to_gene",
)
KIND_ALIASES = (
    "kind", "interaction_kind", "spatial_kind", "support_kind",
)
FAMILY_ALIASES = (
    "family", "interaction_family", "pathway", "pathway_name",
)
DB_ALIASES = (
    "database", "source_db", "resource", "evidence_source",
)


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def _decode(xs):
    out=[]
    for x in xs:
        if isinstance(x,(bytes,np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return tuple(out)


def read_10x_cells_by_genes(path: str | Path):
    """Read the complete 10x matrix as cells x genes CSR."""
    path=Path(path)
    with h5py.File(path,"r") as h5:
        g=h5["matrix"]
        data=np.asarray(g["data"])
        indices=np.asarray(g["indices"])
        indptr=np.asarray(g["indptr"])
        shape=tuple(int(x) for x in np.asarray(g["shape"]))
        csc=sparse.csc_matrix((data,indices,indptr),shape=shape)
        barcodes=_decode(np.asarray(g["barcodes"]))
        fg=g["features"]
        if "name" in fg:
            genes=_decode(np.asarray(fg["name"]))
        elif "id" in fg:
            genes=_decode(np.asarray(fg["id"]))
        else:
            raise ValueError("10x H5 features group lacks name/id")
    if csc.shape != (len(genes),len(barcodes)):
        raise ValueError("10x matrix shape does not match features/barcodes")
    return csc.T.tocsr(),barcodes,genes


def normalized_expression_dense(X: sparse.csr_matrix) -> np.ndarray:
    """
    Library-size normalization followed by log1p.

    The target library is the median positive measured library size rather than
    a fixed external count.  All measured genes remain present.
    """
    X=sparse.csr_matrix(X,dtype=np.float64)
    lib=np.asarray(X.sum(axis=1)).ravel()
    pos=lib[lib>0]
    target=float(np.median(pos)) if len(pos) else 1.0
    scale=np.divide(target,lib,out=np.zeros_like(lib),where=lib>0)
    Y=X.multiply(scale[:,None]).toarray()
    np.log1p(Y,out=Y)
    return Y.astype(np.float32,copy=False)


def discover_functional_gmts(resources_root: str | Path) -> list[Path]:
    root=Path(resources_root)
    if not root.exists():
        return []
    out=[]
    for p in root.rglob("*.gmt"):
        name=p.name.lower()
        if any(t in name for t in SIGNALING_NAME_TOKENS):
            continue
        if any(t in name for t in FUNCTIONAL_NAME_TOKENS):
            out.append(p.resolve())
    return sorted(set(out))


def parse_gmt(path: str | Path):
    """Yield (set_name, genes) from a standard GMT file."""
    with open(path,"r",encoding="utf-8",errors="replace") as f:
        for line in f:
            parts=line.rstrip("\n").split("\t")
            if len(parts)<3:
                continue
            genes=[g.strip() for g in parts[2:] if g.strip()]
            if genes:
                yield parts[0],genes


def functional_prior(
    genes: tuple[str,...],
    gmt_paths: list[Path],
):
    """
    Build Eq. 13-15 style panel-projected functional prior.

    Each gene set contributes a size-normalized clique.  Each resource layer is
    then divided by its median positive weighted degree before equal averaging,
    preventing a larger file from dominating merely by record count.

    The normalized Laplacian is zero on genes with no prior degree, so the
    identity term remains their complete molecular weight.
    """
    G=len(genes)
    idx={str(g):i for i,g in enumerate(genes)}
    layers=[]
    audits=[]

    for path in gmt_paths:
        W=np.zeros((G,G),dtype=np.float64)
        n_sets=0
        n_supported=0
        represented=set()
        for _,members in parse_gmt(path):
            n_sets+=1
            ids=sorted({idx[g] for g in members if g in idx})
            k=len(ids)
            if k<2:
                continue
            n_supported+=1
            represented.update(ids)
            w=1.0/(k-1.0)
            q=np.asarray(ids,dtype=np.int64)
            W[np.ix_(q,q)] += w
            W[q,q] -= w

        deg=W.sum(axis=1)
        pos=deg[deg>0]
        scale=float(np.median(pos)) if len(pos) else np.nan
        if np.isfinite(scale) and scale>0:
            W/=scale
            layers.append(W)
            layer_used=True
        else:
            layer_used=False

        audits.append({
            "path":str(path),
            "sha256":sha256_file(path),
            "n_sets":n_sets,
            "n_panel_supported_sets":n_supported,
            "n_panel_genes_represented":len(represented),
            "median_positive_degree_before_normalization":(
                None if not np.isfinite(scale) else scale
            ),
            "used":layer_used,
        })

    if layers:
        W=sum(layers)/len(layers)
    else:
        W=np.zeros((G,G),dtype=np.float64)

    deg=W.sum(axis=1)
    inv=np.zeros(G,dtype=np.float64)
    q=deg>0
    inv[q]=1.0/np.sqrt(deg[q])
    # L = D^-1/2 (D-W) D^-1/2; isolated genes get zero rows/cols.
    L=np.diag(q.astype(np.float64)) - (inv[:,None]*W)*inv[None,:]
    L=(L+L.T)*0.5

    return W,L,audits


def contact_edge_table(interfaces: pd.DataFrame) -> pd.DataFrame:
    q=interfaces.cell_j_index.notna()
    E=interfaces.loc[q,[
        "interface_id","cell_i_index","cell_j_index",
        "tension","delta_pressure","owner_patch_id"
    ]].copy()
    E["cell_i_index"]=E.cell_i_index.astype(np.int64)
    E["cell_j_index"]=E.cell_j_index.astype(np.int64)
    E["interface_id"]=E.interface_id.astype(np.int64)
    return E.reset_index(drop=True)


def molecular_edge_distance(
    Y: np.ndarray,
    edges: pd.DataFrame,
    L: np.ndarray,
    lam: float=1.0,
    batch_size: int=4096,
) -> np.ndarray:
    """
    d_g(i,j)^2 = d^T (I + lambda L) d on measured contact support.

    Dense batching is deliberate: G is only hundreds for current Xenium panels
    and this avoids Python loops over ~10^5 interfaces.
    """
    if lam<0:
        raise ValueError("lambda must be nonnegative")
    i=edges.cell_i_index.to_numpy(np.int64)
    j=edges.cell_j_index.to_numpy(np.int64)
    out=np.empty(len(edges),dtype=np.float64)
    for a in range(0,len(edges),batch_size):
        b=min(len(edges),a+batch_size)
        D=Y[i[a:b]].astype(np.float64)-Y[j[a:b]].astype(np.float64)
        ident=np.einsum("ij,ij->i",D,D,optimize=True)
        if lam>0 and np.any(L):
            LD=D@L
            prior=np.einsum("ij,ij->i",D,LD,optimize=True)
        else:
            prior=np.zeros(b-a,dtype=np.float64)
        out[a:b]=np.sqrt(np.maximum(0.0,ident+lam*prior))
    return out


def robust_location_scale(x, eps=1e-12):
    x=np.asarray(x,dtype=np.float64)
    x=x[np.isfinite(x)]
    if len(x)==0:
        return {"n":0,"median":None,"q25":None,"q75":None,"iqr":None,"scale_valid":False}
    q25,med,q75=np.quantile(x,[.25,.5,.75])
    iqr=float(q75-q25)
    return {
        "n":int(len(x)),
        "median":float(med),
        "q25":float(q25),
        "q75":float(q75),
        "iqr":iqr,
        "scale_valid":bool(np.isfinite(iqr) and iqr>eps),
    }


def robust_standardize(x, summary, eps=1e-12):
    x=np.asarray(x,dtype=np.float64)
    out=np.full_like(x,np.nan)
    if not summary.get("scale_valid",False):
        return out
    q=np.isfinite(x)
    out[q]=(x[q]-float(summary["median"]))/(float(summary["iqr"])+eps)
    return out


def mechanical_relations(edges: pd.DataFrame):
    """
    Preserve primitive mechanics and missingness.

    The preflight intentionally does not assign a biological sign to tension
    magnitude.  It merely freezes robust Level-0 scales for certified T and
    delta-p separately.
    """
    tau=edges.tension.to_numpy(np.float64)
    dp=edges.delta_pressure.to_numpy(np.float64)
    st=robust_location_scale(tau)
    sd=robust_location_scale(dp)
    return st,sd,robust_standardize(tau,st),robust_standardize(dp,sd)


def _pick(cols,aliases):
    lower={str(c).lower():c for c in cols}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def discover_signaling_tables(resources_root: str | Path) -> list[Path]:
    root=Path(resources_root)
    if not root.exists():
        return []
    out=[]
    for ext in ("*.csv","*.tsv","*.parquet"):
        for p in root.rglob(ext):
            name=p.name.lower()
            if any(t in name for t in SIGNALING_NAME_TOKENS):
                out.append(p.resolve())
    return sorted(set(out))


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower()==".parquet":
        return pd.read_parquet(path)
    sep="\t" if path.suffix.lower()==".tsv" else ","
    return pd.read_csv(path,sep=sep,low_memory=False)


def audit_signaling_resources(
    resources_root: str | Path,
    panel_genes: set[str],
):
    """
    Audit only.  v0.7.1 does not guess interaction family/spatial type.

    A table is hierarchy-ready only if sender, receiver, and explicit spatial
    kind columns exist.  Unknown kind values are not coerced.
    """
    candidates=discover_signaling_tables(resources_root)
    tables=[]
    canonical={}
    allowed_kinds={"contact","diffusible"}

    for p in candidates:
        rec={
            "path":str(p),
            "sha256":sha256_file(p),
            "status":"UNREAD",
        }
        try:
            df=_read_table(p)
            s=_pick(df.columns,SENDER_ALIASES)
            r=_pick(df.columns,RECEIVER_ALIASES)
            k=_pick(df.columns,KIND_ALIASES)
            fam=_pick(df.columns,FAMILY_ALIASES)
            db=_pick(df.columns,DB_ALIASES)

            rec.update({
                "n_rows":int(len(df)),
                "sender_column":None if s is None else str(s),
                "receiver_column":None if r is None else str(r),
                "kind_column":None if k is None else str(k),
                "family_column":None if fam is None else str(fam),
                "database_column":None if db is None else str(db),
            })
            if s is None or r is None:
                rec["status"]="NO_DIRECTED_SCHEMA"
                tables.append(rec);continue
            if k is None:
                rec["status"]="UNTYPED_SPATIAL_SUPPORT"
                tables.append(rec);continue

            n_valid=0;n_panel=0;bad_kinds=set()
            for row in df.itertuples(index=False,name=None):
                # itertuples doesn't map by column name cleanly here; use iloc
                pass
            # vectorized extraction
            a=df[s].astype(str).str.strip()
            b=df[r].astype(str).str.strip()
            kind=df[k].astype(str).str.strip().str.lower()
            valid=(a!="")&(b!="")
            bad=set(kind[valid].unique())-allowed_kinds
            bad_kinds={str(x) for x in bad}
            if bad_kinds:
                rec["status"]="UNKNOWN_SPATIAL_KIND"
                rec["unknown_kinds"]=sorted(bad_kinds)
                tables.append(rec);continue

            for idx in np.flatnonzero(valid.to_numpy()):
                aa=str(a.iloc[idx]);bb=str(b.iloc[idx]);kk=str(kind.iloc[idx])
                n_valid+=1
                if aa in panel_genes and bb in panel_genes:
                    n_panel+=1
                family="unspecified" if fam is None else str(df.iloc[idx][fam])
                source=p.stem if db is None else str(df.iloc[idx][db])
                key=(aa,bb,kk)
                x=canonical.setdefault(key,{
                    "sender_gene":aa,
                    "receiver_gene":bb,
                    "kind":kk,
                    "families":set(),
                    "sources":set(),
                    "evidence_rows":0,
                })
                x["families"].add(family)
                x["sources"].add(source)
                x["evidence_rows"]+=1

            rec["status"]="PARSED_TYPED"
            rec["n_directed_rows"]=n_valid
            rec["n_panel_supported_rows"]=n_panel
        except Exception as exc:
            rec["status"]="ERROR"
            rec["error"]=f"{type(exc).__name__}: {exc}"
        tables.append(rec)

    registry=[]
    for key,x in sorted(canonical.items()):
        registry.append({
            "sender_gene":x["sender_gene"],
            "receiver_gene":x["receiver_gene"],
            "kind":x["kind"],
            "families":"|".join(sorted(x["families"])),
            "sources":"|".join(sorted(x["sources"])),
            "evidence_rows":int(x["evidence_rows"]),
            "panel_supported":bool(
                x["sender_gene"] in panel_genes and
                x["receiver_gene"] in panel_genes
            ),
        })

    reg=pd.DataFrame(registry)
    typed_tables=sum(r["status"]=="PARSED_TYPED" for r in tables)
    panel_supported=int(reg.panel_supported.sum()) if len(reg) else 0
    status="PASS" if typed_tables>0 and panel_supported>0 else "HOLD"
    return {
        "status":status,
        "n_candidate_tables":len(candidates),
        "n_typed_tables":typed_tables,
        "n_canonical_interactions":int(len(reg)),
        "n_panel_supported_interactions":panel_supported,
        "tables":tables,
    },reg


def molecular_prior_status(audits):
    used=sum(bool(x["used"]) for x in audits)
    represented=max((x["n_panel_genes_represented"] for x in audits),default=0)
    return {
        "status":"PASS" if used>0 else "HOLD",
        "n_resource_layers_used":used,
        "max_panel_genes_represented_in_one_layer":represented,
    }


__all__=[
    "sha256_file",
    "read_10x_cells_by_genes",
    "normalized_expression_dense",
    "discover_functional_gmts",
    "functional_prior",
    "contact_edge_table",
    "molecular_edge_distance",
    "robust_location_scale",
    "robust_standardize",
    "mechanical_relations",
    "audit_signaling_resources",
    "molecular_prior_status",
]
