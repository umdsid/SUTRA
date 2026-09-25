#!/usr/bin/env python3
from pathlib import Path
import argparse,numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family":"Arial","font.size":8,"axes.linewidth":.75,"pdf.fonttype":42,"ps.fonttype":42})
SUT="selected__SUTRA_integrated"
METHODS=[("Integrated SUTRA",SUT),("Molecular-only","selected__SUTRA_molecular_only"),("PCA","selected__PCA"),("kPCA*","selected__kPCAstar"),("UMAP","selected__UMAP"),("t-SNE","selected__t_SNE")]
# A/B intentionally show five NONTRIVIAL SUTRA-vs-comparator maps. SUTRA-vs-SUTRA is identically zero.
MAPS=METHODS[1:]
CH=[("molecular_distance","Molecular\ndistance\n(↓ closer)"),("mechanics_effective","Mechanics\n(↓ lower cost)"),("communication_support_effective","Communication\n(↑ more support)"),("geometry_effective","Geometry\n(↓ lower cost)")]

def inc(n,uv,m):
 z=np.zeros(n,np.int32);e=uv[m]
 if len(e):np.add.at(z,e[:,0],1);np.add.at(z,e[:,1],1)
 return z
def localD(n,uv,a,b):
 sh=inc(n,uv,a&b);so=inc(n,uv,a&~b);co=inc(n,uv,~a&b);den=sh+so+co
 return np.divide(so-co,den,out=np.full(n,np.nan),where=den>0),den
def dense_supported(xy,D,sup,nx,ny):
 x,y=xy[:,0],xy[:,1];xe=np.linspace(x.min(),x.max(),nx+1);ye=np.linspace(y.min(),y.max(),ny+1)
 ix=np.clip(np.searchsorted(xe,x,side="right")-1,0,nx-1);iy=np.clip(np.searchsorted(ye,y,side="right")-1,0,ny-1)
 X=[];Y=[];V=[]
 for j in range(ny):
  for i in range(nx):
   m=(ix==i)&(iy==j)&np.isfinite(D)
   if m.sum()>=5 and sup[m].sum()>=8:
    med=float(np.nanmedian(D[m]));ids=np.flatnonzero(m);X.extend(x[ids]);Y.extend(y[ids]);V.extend([med]*len(ids))
 return np.asarray(X),np.asarray(Y),np.asarray(V)
def shift(sel,univ):
 q=np.nanquantile(univ,[.25,.5,.75]);return (np.nanmedian(sel)-q[1])/max(q[2]-q[0],1e-12)
def ecdf_all(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)];x=np.sort(x);return x,np.arange(1,len(x)+1)/len(x)
def head(fig,y,L,t):
 fig.text(.018,y,L,fontsize=18,fontweight="bold",ha="left",va="top");fig.text(.065,y,t,fontsize=13,fontweight="bold",ha="left",va="top")

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",default=str(Path.home()/"Desktop/SUTRA"));a=ap.parse_args();root=Path(a.root).expanduser()
 src=root/"results/suppfig2_fair_matched_candidate_v152";out=root/"figures/suppfig2_v19_10_D_layout_repair";out.mkdir(parents=True,exist_ok=True)
 dat={}
 for s,o in [("healthy_reference","Brain"),("nondiseased_kidney","Kidney")]:
  dat[o]=(pd.read_parquet(src/s/"aligned_cells_xy.parquet"),pd.read_parquet(src/s/"matched_candidate_edge_table.parquet"))

 # Landscape canvas matches the approved reference proportions; manual axes lock geometry.
 fig=plt.figure(figsize=(15.2,11.7),facecolor="white")
 head(fig,.985,"A","Brain: local selection difference")
 head(fig,.735,"B","Kidney: local selection difference")
 head(fig,.480,"C","Multichannel selection profiles")
 head(fig,.205,"D","Quantitative summaries")

 # Five comparator maps in ONE line per organ.
 xs=np.linspace(.055,.815,5);mw=.155
 mapnorm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1);im=None
 for organ,y0 in [("Brain",.790),("Kidney",.565)]:
  cells,e=dat[organ];xy=cells[["x","y"]].to_numpy(float);uv=e[["u","v"]].to_numpy(int);sut=e[SUT].astype(bool).to_numpy()
  nx,ny=(48,32) if organ=="Brain" else (72,32)
  for x0,(name,col) in zip(xs,MAPS):
   ax=fig.add_axes([x0,y0,mw,.135 if organ=="Brain" else .105])
   D,sup=localD(len(cells),uv,sut,e[col].astype(bool).to_numpy());X,Y,V=dense_supported(xy,D,sup,nx,ny)
   im=ax.scatter(X,Y,c=V,s=2.0 if organ=="Brain" else .85,cmap="coolwarm",norm=mapnorm,linewidths=0,alpha=.92,rasterized=True)
   ax.set_title(name,fontsize=9,pad=2);ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
   for sp in ax.spines.values():sp.set_visible(False)

 # ONE simple shared colorbar, as in approved reference. Meaning text belongs in caption.
 cax=fig.add_axes([.265,.525,.470,.014]);cb=fig.colorbar(im,cax=cax,orientation="horizontal")
 cb.set_ticks([-1,0,1]);cb.set_ticklabels(["−1","0","+1"]);cb.ax.tick_params(labelsize=8,pad=2,length=3)
 fig.text(.500,.548,"Comparator-only dominated  ←  Local relation-selection balance  →  SUTRA-only dominated",ha="center",va="bottom",fontsize=7.8)

 # C: preserve successful two-matrix format.
 haxes=[fig.add_axes([.095,.285,.375,.155]),fig.add_axes([.565,.285,.355,.155])]
 vals={}
 for o,(cells,e) in dat.items():
  vals[o]=np.array([[shift(e.loc[e[c].astype(bool),q].to_numpy(float),e[q].to_numpy(float)) for q,_ in CH] for _,c in METHODS])
 him=None
 for ax,o in zip(haxes,["Brain","Kidney"]):
  v=vals[o];him=ax.imshow(v,aspect="auto",cmap="coolwarm",norm=TwoSlopeNorm(vmin=-.8,vcenter=0,vmax=.8))
  ax.set_title(o,fontsize=11.5,fontweight="bold",pad=5);ax.set_yticks(range(6),[m[0] for m in METHODS],fontsize=7.5);ax.set_xticks(range(4),[q[1] for q in CH],fontsize=7.2)
  for iy in range(6):
   for ix in range(4):ax.text(ix,iy,f"{v[iy,ix]:+.2f}",ha="center",va="center",fontsize=6.5)
 hc=fig.add_axes([.936,.285,.010,.155]);hcb=fig.colorbar(him,cax=hc);hcb.set_ticks([-.5,0,.5]);hcb.ax.tick_params(labelsize=7)

 # D-left: expose the coverage effect rather than hiding it.
 # For each comparator, show global Jaccard and Jaccard restricted to candidate
 # edges whose endpoints are covered by both SUTRA and that comparator.
 ax=fig.add_axes([.050,.030,.455,.125])
 x=np.arange(5); organ_offsets={"Brain":-.19,"Kidney":.19}; bw=.16
 for o in ["Brain","Kidney"]:
  e=dat[o][1]; uv=e[["u","v"]].to_numpy(int); n=len(dat[o][0])
  sut=e[SUT].astype(bool).to_numpy(); cs=inc(n,uv,sut)>0
  vg=[]; vc=[]
  for _,c in MAPS:
   cmp=e[c].astype(bool).to_numpy(); cc=inc(n,uv,cmp)>0
   union=sut|cmp; inter=sut&cmp
   vg.append(inter.sum()/max(union.sum(),1))
   common=cs&cc
   edge_common=common[uv[:,0]]&common[uv[:,1]]
   uc=union&edge_common; ic=inter&edge_common
   vc.append(ic.sum()/max(uc.sum(),1))
  center=x+organ_offsets[o]
  # Global = filled; common-coverage = white fill with matching edge color.
  color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0 if o=="Brain" else 1]
  b1=ax.bar(center-bw/2,vg,bw,color=color,label=f"{o}: global")
  b2=ax.bar(center+bw/2,vc,bw,facecolor="white",edgecolor=color,linewidth=1.2,
            label=f"{o}: common coverage")
  for bar,v in zip(b1,vg):
   ax.text(bar.get_x()+bar.get_width()/2,v+.018,f"{v:.2f}",ha="center",va="bottom",fontsize=5.8)
  for bar,v in zip(b2,vc):
   ax.text(bar.get_x()+bar.get_width()/2,v+.018,f"{v:.2f}",ha="center",va="bottom",fontsize=5.8)
 ax.set_ylim(0,.92)
 ax.set_ylabel("Jaccard with integrated SUTRA",fontsize=7.5)
 ax.set_xticks(x,[m[0] for m in MAPS],fontsize=7.2)
 ax.set_title("Relation overlap with integrated SUTRA",fontsize=9.0,fontweight="bold",pad=18)
 ax.legend(frameon=False,ncol=4,fontsize=5.8,loc="upper left",bbox_to_anchor=(0.0,1.08),borderaxespad=0.0,columnspacing=.9,handlelength=1.5)
 ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)

 # D-right: restore the exact APPROVED visual behavior:
 # full ECDF including zeros + symlog x scale. This creates the separated sigmoidal curves
 # seen in the reference while retaining the raw composite_merge_cost values.
 ax=fig.add_axes([.565,.030,.305,.125]);colors=plt.rcParams["axes.prop_cycle"].by_key()["color"];mh=[]
 allpos=[]
 for o in ["Brain","Kidney"]:
  e=dat[o][1]
  for _,c in METHODS:
   z=e.loc[e[c].astype(bool),"composite_merge_cost"].to_numpy(float);allpos.extend(z[np.isfinite(z)&(z>0)])
 linthresh=max(np.nanpercentile(np.asarray(allpos),1),1e-4)
 for mi,(name,c) in enumerate(METHODS):
  for o,ls in [("Brain","-"),("Kidney","--")]:
   e=dat[o][1];xx,yy=ecdf_all(e.loc[e[c].astype(bool),"composite_merge_cost"].to_numpy(float))
   ax.plot(xx,yy,ls=ls,color=colors[mi],lw=1.45)
  mh.append(Line2D([0],[0],color=colors[mi],lw=1.5,label=name))
 ax.set_xscale("symlog",linthresh=linthresh,linscale=1.0);ax.set_ylim(0,1.02)
 ax.set_xlabel("Composite merge cost",fontsize=7.5);ax.set_ylabel("Cumulative fraction",fontsize=7.5)
 leg=ax.legend(handles=mh,frameon=False,fontsize=6.6,loc="upper left",bbox_to_anchor=(1.02,1.02),borderaxespad=0);ax.add_artist(leg)
 ax.legend(handles=[Line2D([0],[0],color=".15",ls="-",label="Brain (solid)"),Line2D([0],[0],color=".15",ls="--",label="Kidney (dashed)")],frameon=False,fontsize=6.7,loc="lower left",bbox_to_anchor=(1.02,.02),borderaxespad=0)
 ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)

 # Node-coverage/singleton audit. This does not alter the figure or selections.
 rows=[]
 for organ,(cells,e) in dat.items():
  uv=e[["u","v"]].to_numpy(int); n=len(cells)
  for name,col in METHODS:
   mask=e[col].astype(bool).to_numpy()
   deg=inc(n,uv,mask)
   rows.append({
    "organ":organ,"method":name,"n_cells":n,
    "n_selected_edges":int(mask.sum()),
    "n_singletons":int((deg==0).sum()),
    "singleton_fraction":float((deg==0).mean()),
    "n_cells_covered":int((deg>0).sum()),
    "cell_coverage_fraction":float((deg>0).mean()),
    "selected_degree_mean":float(deg.mean()),
    "selected_degree_median":float(np.median(deg)),
    "selected_degree_p90":float(np.quantile(deg,.90)),
    "selected_degree_p99":float(np.quantile(deg,.99)),
    "selected_degree_max":int(deg.max()),
   })
 audit=pd.DataFrame(rows)
 audit.to_csv(out/"singleton_node_coverage_audit.csv",index=False)

 # Coverage-aware robustness audit without altering any frozen selection:
 # compute Jaccard (i) globally and (ii) on the induced candidate-edge set whose
 # endpoints are both covered by BOTH SUTRA and the comparator.
 covrows=[]
 for organ,(cells,e) in dat.items():
  uv=e[["u","v"]].to_numpy(int); n=len(cells)
  sut=e[SUT].astype(bool).to_numpy()
  ds=inc(n,uv,sut); cs=ds>0
  for name,col in METHODS[1:]:
   cmp=e[col].astype(bool).to_numpy()
   dc=inc(n,uv,cmp); cc=dc>0
   common_nodes=cs & cc
   edge_common=common_nodes[uv[:,0]] & common_nodes[uv[:,1]]
   union=sut|cmp; inter=sut&cmp
   global_j=float(inter.sum()/max(union.sum(),1))
   union_c=union & edge_common; inter_c=inter & edge_common
   common_j=float(inter_c.sum()/max(union_c.sum(),1))
   covrows.append({
    "organ":organ,"comparator":name,
    "sutra_coverage_fraction":float(cs.mean()),
    "comparator_coverage_fraction":float(cc.mean()),
    "common_covered_cells":int(common_nodes.sum()),
    "common_covered_fraction":float(common_nodes.mean()),
    "global_jaccard":global_j,
    "jaccard_within_common_covered_nodes":common_j,
    "n_union_edges_within_common_covered_nodes":int(union_c.sum()),
    "n_shared_edges_within_common_covered_nodes":int(inter_c.sum())
   })
 cov=pd.DataFrame(covrows)
 cov.to_csv(out/"coverage_aware_overlap_audit.csv",index=False)
 with open(out/"coverage_aware_overlap_audit.txt","w") as f:
  f.write("Coverage-aware overlap sensitivity audit\\n")
  f.write("Frozen selections are NOT changed.\\n")
  f.write("The restricted Jaccard is computed only on candidate edges whose two endpoints are covered by both SUTRA and the comparator.\\n")
  f.write("This is a conservative sensitivity analysis, not an exact node-coverage-matched reselection.\\n\\n")
  f.write(cov.to_string(index=False))
  f.write("\\n")
 print("\\nCOVERAGE-AWARE OVERLAP AUDIT")
 print(cov.to_string(index=False))

 with open(out/"singleton_node_coverage_audit.txt","w") as f:
  f.write("Singleton/node-coverage audit for frozen v1.5.2 selections\\n")
  f.write("All methods use the same physical candidate-edge universe and identical global edge budget.\\n")
  f.write("Equal edge budget does not force equal node coverage; this audit measures that explicitly.\\n\\n")
  f.write(audit.to_string(index=False))
  f.write("\\n")
 print("\\nSINGLETON / NODE-COVERAGE AUDIT")
 print(audit.to_string(index=False))

 fig.savefig(out/"suppfig2_v19_10_D_layout_repair.png",dpi=450,facecolor="white")
 fig.savefig(out/"suppfig2_v19_10_D_layout_repair.pdf",facecolor="white")
 plt.close(fig);print("WROTE",out)
if __name__=="__main__":main()
