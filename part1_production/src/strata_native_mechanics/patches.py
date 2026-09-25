from __future__ import annotations
from collections import defaultdict, deque
import pandas as pd


def contact_graph(interfaces: pd.DataFrame):
    g=defaultdict(set)
    for r in interfaces.itertuples():
        if r.kind!="cell_cell":
            continue
        a,b=int(r.cell_i),int(r.cell_j)
        g[a].add(b);g[b].add(a)
    return g


def partition_core_cells(interfaces: pd.DataFrame, patch_size: int = 300):
    g=contact_graph(interfaces)
    unassigned=set(g.keys())
    patches=[]
    while unassigned:
        seed=min(unassigned)
        q=deque([seed]);core=[]
        unassigned.remove(seed)
        while q and len(core)<patch_size:
            u=q.popleft();core.append(u)
            for v in sorted(g[u]):
                if v in unassigned and len(core)<patch_size:
                    unassigned.remove(v);q.append(v)
        patches.append(core)
    return patches


def add_halo(core, interfaces: pd.DataFrame, hops: int = 1):
    g=contact_graph(interfaces)
    cells=set(core)
    frontier=set(core)
    for _ in range(hops):
        nxt=set()
        for u in frontier:
            nxt.update(g.get(u,()))
        nxt-=cells
        cells|=nxt
        frontier=nxt
    return sorted(cells)
