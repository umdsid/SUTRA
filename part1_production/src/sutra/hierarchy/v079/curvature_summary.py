"""STRATA v0.7.9 loop-normalized holonomy summaries.

No smooth curvature tensor is inferred.

For a spatially nondegenerate resolved loop gamma, define the operational
holonomy density

    kappa_gamma = theta_gamma / A_gamma,

where theta_gamma is the certified v0.7.8 spectral holonomy-angle summary and
A_gamma is the physical polygon area of the ordered Level-0 tissue loop.

This quantity has units angle / spatial-area. It is a discrete connection
observable, not an estimator of Gaussian, Ricci, scalar, or flag curvature.

Overlapping loops are not treated as independent biological replicates.
The release therefore also aggregates loops to cells and transport connected
components.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


X_CANDIDATES=(
    "centroid_x","x_centroid","center_x","x_center",
    "x_location","x_coord","x",
)
Y_CANDIDATES=(
    "centroid_y","y_centroid","center_y","y_center",
    "y_location","y_coord","y",
)


def resolve_xy_columns(cells):
    cols=set(map(str,cells.columns))
    x=next((c for c in X_CANDIDATES if c in cols),None)
    y=next((c for c in Y_CANDIDATES if c in cols),None)
    if x is None or y is None:
        raise RuntimeError(
            "Could not resolve spatial centroid columns. "
            f"Tried X={X_CANDIDATES}, Y={Y_CANDIDATES}; "
            f"available={list(cells.columns)}"
        )
    return x,y


def parse_loop_nodes(s):
    if isinstance(s,(tuple,list,np.ndarray)):
        return tuple(int(x) for x in s)
    return tuple(int(x) for x in str(s).split(";") if str(x)!="")


def polygon_geometry(xy):
    """Ordered polygon area, perimeter and compactness."""
    P=np.asarray(xy,dtype=np.float64)
    if P.ndim!=2 or P.shape[1]!=2 or len(P)<3:
        raise ValueError("xy must be n>=3 by 2")

    x=P[:,0]; y=P[:,1]
    area=.5*abs(float(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1))))
    D=np.roll(P,-1,axis=0)-P
    lengths=np.linalg.norm(D,axis=1)
    perimeter=float(np.sum(lengths))
    edge_rms=float(np.sqrt(np.mean(lengths*lengths)))

    compactness=(
        float(4.0*np.pi*area/(perimeter*perimeter))
        if perimeter>0 else np.nan
    )

    # Scale-aware degeneracy rule: area must exceed roundoff on the squared
    # loop spatial scale. This is numerical, not a biological threshold.
    spatial_scale=max(edge_rms,1.0)
    area_tol=float(
        256.0*np.finfo(np.float64).eps*spatial_scale*spatial_scale
    )
    nondegenerate=bool(np.isfinite(area) and area>area_tol and perimeter>0)

    return {
        "spatial_area":area,
        "spatial_perimeter":perimeter,
        "spatial_edge_rms":edge_rms,
        "spatial_compactness":compactness,
        "spatial_area_tolerance":area_tol,
        "spatial_nondegenerate":nondegenerate,
    }


def attach_spatial_geometry(loop_df,cells,xcol,ycol):
    xy=cells[[xcol,ycol]].to_numpy(np.float64)
    rows=[]
    for r in loop_df.itertuples(index=False):
        d=r._asdict()
        nodes=parse_loop_nodes(d["loop_nodes"])
        if any(u<0 or u>=len(xy) for u in nodes):
            raise IndexError(f"loop contains invalid cell index: {nodes}")
        g=polygon_geometry(xy[list(nodes)])
        d.update(g)
        d["loop_node_count"]=int(len(nodes))
        rows.append(d)
    return pd.DataFrame(rows)


def attach_holonomy_density(df):
    out=df.copy()

    if "spectral_angle_rms" not in out.columns:
        raise RuntimeError("spectral_angle_rms missing")
    if "spectral_angle_max" not in out.columns:
        raise RuntimeError("spectral_angle_max missing")

    eligible=(
        out.fully_resolved.astype(bool)
        & out.spatial_nondegenerate.astype(bool)
        & np.isfinite(out.spectral_angle_rms)
    )

    out["holonomy_density_rms"]=np.nan
    out["holonomy_density_max"]=np.nan
    out["identity_deviation_density"]=np.nan

    out.loc[eligible,"holonomy_density_rms"]=(
        out.loc[eligible,"spectral_angle_rms"]
        /out.loc[eligible,"spatial_area"]
    )
    out.loc[eligible,"holonomy_density_max"]=(
        out.loc[eligible,"spectral_angle_max"]
        /out.loc[eligible,"spatial_area"]
    )
    out.loc[eligible,"identity_deviation_density"]=(
        out.loc[eligible,"identity_deviation_normalized"]
        /out.loc[eligible,"spatial_area"]
    )

    out["directional_resolved_loop"]=(
        out.fully_resolved.astype(bool)
        &(out.n_directional_edges.astype(int)>0)
    )
    out["density_defined"]=eligible
    return out


def loop_transport_component(loop_nodes,node_components):
    nodes=parse_loop_nodes(loop_nodes)
    c=np.asarray(node_components,dtype=np.int64)[list(nodes)]
    if len(c)==0:
        return -1
    u=np.unique(c)
    return int(u[0]) if len(u)==1 else -1


def attach_components(df,node_components):
    out=df.copy()
    out["transport_component"]=[
        loop_transport_component(s,node_components)
        for s in out.loop_nodes
    ]
    out["single_transport_component"]=out.transport_component>=0
    return out


def summarize_loops(df):
    resolved=df[df.fully_resolved.astype(bool)]
    defined=resolved[resolved.density_defined.astype(bool)]
    directional=defined[defined.n_directional_edges.astype(int)>0]

    out={
        "n_candidate_loops":int(len(df)),
        "n_resolved_loops":int(len(resolved)),
        "resolved_fraction":float(len(resolved)/len(df)) if len(df) else 0.0,
        "n_density_defined_loops":int(len(defined)),
        "density_defined_fraction_of_resolved":
            float(len(defined)/len(resolved)) if len(resolved) else 0.0,
        "n_directional_density_loops":int(len(directional)),
        "directional_density_fraction_of_defined":
            float(len(directional)/len(defined)) if len(defined) else 0.0,
        "n_spatially_degenerate_resolved_loops":
            int((resolved.spatial_nondegenerate==False).sum()) if len(resolved) else 0,
    }

    if len(directional):
        for col,prefix in [
            ("spectral_angle_rms","angle_rms"),
            ("spatial_area","area"),
            ("spatial_perimeter","perimeter"),
            ("holonomy_density_rms","density_rms"),
            ("holonomy_density_max","density_max"),
        ]:
            x=directional[col].to_numpy(np.float64)
            out[f"{prefix}_median"]=float(np.median(x))
            out[f"{prefix}_q90"]=float(np.quantile(x,.90))
            out[f"{prefix}_q95"]=float(np.quantile(x,.95))
            out[f"{prefix}_q99"]=float(np.quantile(x,.99))
            out[f"{prefix}_max"]=float(np.max(x))

        # Area-weighted aggregate: total integrated angle / total area.
        out["aggregate_angle_per_area"]=float(
            directional.spectral_angle_rms.sum()
            /directional.spatial_area.sum()
        )
    else:
        for prefix in [
            "angle_rms","area","perimeter","density_rms","density_max"
        ]:
            for suffix in ["median","q90","q95","q99","max"]:
                out[f"{prefix}_{suffix}"]=np.nan
        out["aggregate_angle_per_area"]=np.nan

    return out


def cell_aggregate(df,n_cells):
    """Attach loop burden back to Level-0 cells.

    Counts and sums are descriptive incidence summaries. Because loops overlap,
    they are not interpreted as independent observations.
    """
    rows=[{
        "cell_index":i,
        "incident_resolved_loops":0,
        "incident_directional_loops":0,
        "incident_density_defined_loops":0,
        "holonomy_angle_rms_sum":0.0,
        "holonomy_area_sum":0.0,
        "holonomy_density_rms_sum":0.0,
        "holonomy_density_rms_max":0.0,
    } for i in range(n_cells)]

    for r in df.itertuples(index=False):
        nodes=parse_loop_nodes(r.loop_nodes)
        resolved=bool(r.fully_resolved)
        direction=bool(r.directional_resolved_loop)
        defined=bool(r.density_defined)

        for u in nodes:
            z=rows[u]
            if resolved:
                z["incident_resolved_loops"]+=1
            if direction:
                z["incident_directional_loops"]+=1
            if defined:
                z["incident_density_defined_loops"]+=1
            if direction and defined:
                th=float(r.spectral_angle_rms)
                ar=float(r.spatial_area)
                kd=float(r.holonomy_density_rms)
                z["holonomy_angle_rms_sum"]+=th
                z["holonomy_area_sum"]+=ar
                z["holonomy_density_rms_sum"]+=kd
                z["holonomy_density_rms_max"]=max(
                    z["holonomy_density_rms_max"],kd
                )

    out=pd.DataFrame(rows)
    q=out.incident_directional_loops>0
    out["incident_directional_loop_fraction"]=np.divide(
        out.incident_directional_loops,
        out.incident_resolved_loops,
        out=np.zeros(n_cells,dtype=np.float64),
        where=out.incident_resolved_loops>0,
    )
    out["mean_incident_holonomy_density"]=np.divide(
        out.holonomy_density_rms_sum,
        out.incident_directional_loops,
        out=np.full(n_cells,np.nan,dtype=np.float64),
        where=out.incident_directional_loops>0,
    )
    out["area_weighted_incident_holonomy_density"]=np.divide(
        out.holonomy_angle_rms_sum,
        out.holonomy_area_sum,
        out=np.full(n_cells,np.nan,dtype=np.float64),
        where=out.holonomy_area_sum>0,
    )
    out.loc[~q,"holonomy_density_rms_max"]=np.nan
    return out


def component_aggregate(df):
    q=df[
        df.directional_resolved_loop.astype(bool)
        &df.density_defined.astype(bool)
        &df.single_transport_component.astype(bool)
    ].copy()

    if len(q)==0:
        return pd.DataFrame(columns=[
            "transport_component","n_directional_loops","total_loop_area",
            "total_holonomy_angle_rms","aggregate_angle_per_area",
            "density_median","density_q95","density_max"
        ])

    rows=[]
    for comp,g in q.groupby("transport_component",sort=False):
        rows.append({
            "transport_component":int(comp),
            "n_directional_loops":int(len(g)),
            "total_loop_area":float(g.spatial_area.sum()),
            "total_holonomy_angle_rms":float(g.spectral_angle_rms.sum()),
            "aggregate_angle_per_area":float(
                g.spectral_angle_rms.sum()/g.spatial_area.sum()
            ),
            "density_median":float(g.holonomy_density_rms.median()),
            "density_q95":float(g.holonomy_density_rms.quantile(.95)),
            "density_max":float(g.holonomy_density_rms.max()),
        })
    return pd.DataFrame(rows)


def certify_summary(df):
    q=df[df.density_defined.astype(bool)]
    finite=bool(
        len(q)==0
        or np.isfinite(q[
            [
                "spatial_area","spatial_perimeter",
                "holonomy_density_rms","holonomy_density_max",
            ]
        ].to_numpy()).all()
    )
    positive_area=bool(len(q)==0 or (q.spatial_area>0).all())
    identity_zero=True
    z=q[q.n_directional_edges.astype(int)==0]
    if len(z):
        identity_zero=bool(
            (np.abs(z.holonomy_density_rms)<=1e-12).all()
            and (np.abs(z.holonomy_density_max)<=1e-12).all()
        )

    return {
        "all_defined_density_values_finite":finite,
        "all_defined_loop_areas_positive":positive_area,
        "identity_only_density_zero":identity_zero,
        "certificate_pass":bool(finite and positive_area and identity_zero),
    }
