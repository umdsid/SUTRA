from pathlib import Path
import json, shutil
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

HOME=Path.home(); ROOT=HOME/'Desktop'/'SUTRA'; RES=ROOT/'results'
OUTROOT=HOME/'Desktop'/'SUTRA_GITHUB_REPRO_OUTPUT'
O5=OUTROOT/'MainFig5'; O6=OUTROOT/'MainFig6'; O5.mkdir(parents=True,exist_ok=True); O6.mkdir(parents=True,exist_ok=True)
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':0.7,'pdf.fonttype':42,'ps.fonttype':42,'savefig.pad_inches':0.03})
def clean(ax): ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
def save(fig,out,stem):
    fig.savefig(out/f'{stem}.pdf',bbox_inches='tight',facecolor='white'); fig.savefig(out/f'{stem}.png',dpi=450,bbox_inches='tight',facecolor='white'); plt.close(fig)

def first_existing(paths):
    for p in paths:
        if p.exists(): return p
    raise FileNotFoundError('\n'.join(str(x) for x in paths))

# ---------------- Fig 5 ----------------
GW=RES/'Fig5_Disease_Trajectories'/'gw_multistart_dense_v5'/'multistart_scale_summary.csv'
gw=pd.read_csv(GW).sort_values('u_target')
ucol='u_target'; dcol='best_gw_distance'
if not {ucol,dcol}.issubset(gw.columns): raise RuntimeError(f'Fig5 GW schema mismatch: {list(gw.columns)}')
if len(gw)!=25: raise RuntimeError(f'Fig5 GW expected 25 states; found {len(gw)}')
print('PASS — Fig5 GW source/schema preflight')

# A — global relational displacement
fig,ax=plt.subplots(figsize=(3.55,2.55))
ax.plot(gw[ucol],gw[dcol],'-',lw=1.25,zorder=1)
sel=gw[np.isclose(gw[ucol].to_numpy()[:,None],np.array([.40,.46,.48])[None,:],atol=.006).any(axis=1)]
ax.scatter(sel[ucol],sel[dcol],s=25,zorder=3)
for x in [.40,.46,.48]:
    ax.axvline(x,lw=.55,ls=':',alpha=.35,zorder=0)
ax.set_xlim(float(gw[ucol].min()),float(gw[ucol].max()))
ax.set_xlabel('Hierarchy progression, $u$')
ax.set_ylabel('NDK–PRCC relational displacement')
clean(ax)
fig.tight_layout(pad=.35)
save(fig,O5,'panel_A')

# B — evolution of the full object-level distortion distribution
# NOTE: these v7 files contain the historical endpoint-conditioned GW attribution.
# They are not relabelled as the newer NDK-referenced Delta used in panel C.
LOC=RES/'Fig5_Disease_Trajectories'/'structural_trajectory_v7'/'localization'
states=[.08,.16,.24,.32,.40,.48,.56]
rows=[]
for u in states:
    p=LOC/f'prcc_object_distortion_u{u:.2f}.csv'
    d=pd.read_csv(p)
    if not {'gw_distortion','cell_mass'}.issubset(d.columns):
        raise RuntimeError(f'Fig5B bad schema: {p}')
    x=pd.to_numeric(d.gw_distortion,errors='coerce').to_numpy(float)
    w=pd.to_numeric(d.cell_mass,errors='coerce').fillna(0).to_numpy(float)
    ok=np.isfinite(x)&np.isfinite(w)&(w>0); x=x[ok]; w=w[ok]
    if len(x)<100: raise RuntimeError(f'Fig5B too few objects at u={u}')
    ix=np.argsort(x); xx=x[ix]; ww=w[ix]; cw=np.cumsum(ww)/ww.sum()
    def wq(q): return float(xx[min(np.searchsorted(cw,q),len(xx)-1)])
    rows.append((u,wq(.25),wq(.50),wq(.75),wq(.90),wq(.95),wq(.99)))
q=pd.DataFrame(rows,columns=['u','q25','q50','q75','q90','q95','q99'])

fig,ax=plt.subplots(figsize=(4.35,2.85))
ax.fill_between(q.u,q.q25,q.q75,alpha=.16,label='25–75%')
ax.fill_between(q.u,q.q75,q.q95,alpha=.10,label='75–95%')
ax.plot(q.u,q.q50,'o-',lw=1.25,ms=3.2,label='Median')
ax.plot(q.u,q.q95,'o-',lw=1.05,ms=2.8,label='95th percentile')
ax.plot(q.u,q.q99,'o--',lw=.85,ms=2.4,label='99th percentile')
ax.set_xlabel('Hierarchy progression, $u$')
ax.set_ylabel('Endpoint-conditioned relational distortion')
ax.set_xticks(states[::2])
clean(ax)
ax.legend(frameon=False,fontsize=6.3,ncol=2,loc='upper left',
          handlelength=1.5,columnspacing=.9)
fig.tight_layout(pad=.35)
save(fig,O5,'panel_B')

# C exact V6 freeze-candidate: copy, do not recalculate
CROOT=RES/'Main_Fig5_V6'
for ext in ['png','pdf']:
    src=CROOT/f'Fig5C_NDK_reference_deviation_v6.{ext}'
    if not src.exists(): raise FileNotFoundError(src)
    shutil.copy2(src,O5/f'panel_C.{ext}')

# D — complete spatial-territory trajectory structure
# The frozen territory summary contains all 96 territories. It stores three
# independently measured summaries: early mean (u=.08–.24), u=.40, and late
# mean (u=.48–.56). We do not interpolate unobserved territory values.
TP=RES/'Fig5_Disease_Trajectories'/'trajectory_maps_v8'/'territory_trajectory_summary.csv'
t=pd.read_csv(TP)
need={'territory','early_mean_008_024','u040','late_mean_048_056','descriptive_pattern'}
if not need.issubset(t.columns):
    raise RuntimeError(f'Fig5D missing {sorted(need-set(t.columns))}')
if len(t)!=96:
    raise RuntimeError(f'Fig5D expected 96 frozen territories, found {len(t)}')

# Hard source audit. Territories with no finite values in all three displayed
# summaries are retained spatially but are not assigned a quantitative trajectory
# class. Partial missingness remains a hard failure.
value_cols=['early_mean_008_024','u040','late_mean_048_056']
for c in value_cols:
    t[c]=pd.to_numeric(t[c],errors='coerce')

finite=np.isfinite(t[value_cols].to_numpy(float))
nfinite=finite.sum(axis=1)
partial=(nfinite>0)&(nfinite<3)
if partial.any():
    bad=t.loc[partial,['territory','descriptive_pattern']+value_cols]
    bad.to_csv(O5/'panel_D_partial_missing_error.csv',index=False)
    raise RuntimeError(
        'Fig5D has partially observed territories. These cannot be classified as '
        'not evaluable and cannot be imputed. See panel_D_partial_missing_error.csv.'
    )

not_eval=(nfinite==0)
source_classes=t['descriptive_pattern'].astype(str).copy()
audit=t.loc[not_eval,['territory','descriptive_pattern']+value_cols].copy()
audit.insert(0,'source_row',audit.index.astype(int))
audit['final_display_class']='not_evaluable'
audit['reason']='no finite endpoint-conditioned distortion summary at early, u=.40, or late'
audit.to_csv(O5/'panel_D_not_evaluable_territories.csv',index=False)

# Permanent invariant: an all-missing territory cannot retain a quantitative class.
t.loc[not_eval,'descriptive_pattern']='not_evaluable'
quantitative=~not_eval
if t.loc[quantitative,value_cols].isna().any().any():
    raise RuntimeError('Fig5D quantitative territories contain missing values.')

print(f'FIG5D AUDIT — territories={len(t)}, evaluable={int(quantitative.sum())}, '
      f'not_evaluable={int(not_eval.sum())}, quantitative_cells={int(finite[quantitative].sum())}')
if not_eval.any():
    print('FIG5D NOT EVALUABLE:')
    print(audit.to_string(index=False))

# Audit source-label correction rather than silently rewriting provenance.
label_audit=pd.DataFrame({
    'territory':t['territory'],
    'source_class':source_classes,
    'final_display_class':t['descriptive_pattern'].astype(str),
    'evaluable':quantitative
})
label_audit.to_csv(O5/'panel_D_classification_audit.csv',index=False)

order_names=['comparatively_stable','u040_enriched_then_reduced',
             'late_recruitment','progressive_increase']
pretty={'comparatively_stable':'comparatively stable',
        'u040_enriched_then_reduced':r'$u=.40$ enriched → reduced',
        'late_recruitment':'late recruitment',
        'progressive_increase':'progressive increase'}
t['pattern_order']=pd.Categorical(t.descriptive_pattern,categories=order_names,ordered=True)
# Within each descriptive family order by the magnitude and timing of the frozen trajectory.
t['peak']=t[['early_mean_008_024','u040','late_mean_048_056']].max(axis=1)
t=t.sort_values(['pattern_order','peak','territory'],ascending=[True,False,True]).reset_index(drop=True)

M=t[value_cols].to_numpy(float)
if M.shape != (96,3):
    raise RuntimeError(f'Fig5D matrix invariant failed: {M.shape} != (96, 3)')
if len(set(t['territory'].astype(str))) != 96:
    raise RuntimeError('Fig5D territory identity invariant failed: territories are not unique')
row_all_missing=(~np.isfinite(M)).all(axis=1)
row_all_finite=np.isfinite(M).all(axis=1)
if not np.all(row_all_missing | row_all_finite):
    raise RuntimeError('Fig5D post-sort matrix contains partial missingness.')
if not np.array_equal(row_all_missing,(t['descriptive_pattern'].astype(str)=='not_evaluable').to_numpy()):
    raise RuntimeError('Fig5D not-evaluable classification/matrix mask invariant failed.')
vmax=float(np.quantile(M[row_all_finite],.99))
Mplot=np.ma.masked_invalid(M)
print(f'FIG5D POST-SORT AUDIT: PASS — shape=(96,3), '
      f'evaluable={int(row_all_finite.sum())}, not_evaluable={int(row_all_missing.sum())}, '
      '96 unique territories')
matrix_audit=t[['territory','descriptive_pattern']+value_cols].copy()
matrix_audit.insert(0,'render_row',np.arange(len(matrix_audit),dtype=int))
matrix_audit['evaluable']=matrix_audit['descriptive_pattern'].astype(str)!='not_evaluable'
matrix_audit.to_csv(O5/'panel_D_render_matrix_96x3.csv',index=False)

fig=plt.figure(figsize=(4.85,3.45))
gs=fig.add_gridspec(1,2,width_ratios=[4.5,1.15],wspace=.12)
ax=fig.add_subplot(gs[0,0])
cmap=plt.get_cmap('viridis').copy()
cmap.set_bad('0.82')
im=ax.imshow(Mplot,aspect='auto',interpolation='none',origin='upper',
             extent=(-.5,2.5,95.5,-.5),
             vmin=0,vmax=vmax,cmap=cmap,rasterized=True)
ax.set_xlim(-.5,2.5)
ax.set_ylim(95.5,-.5)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['Early\n.08–.24',r'$u=.40$','Late\n.48–.56'])
ax.set_ylabel('Spatial territories')
ax.set_xlabel('Hierarchy interval')
# Boundaries and family labels.
bounds=[]; pos=0
for name in order_names:
    n=int((t.descriptive_pattern==name).sum())
    if n:
        bounds.append((name,pos,n))
        pos+=n
ax.set_yticks([])
clean(ax)

# IMPORTANT: do not share y with the image axis.  A shared annotation axis can
# autoscale the raster when external brackets/text extend past row centres,
# producing a false white horizontal band at a group boundary.
ax.set_ylim(len(t)-.5,-.5)
axr=fig.add_subplot(gs[0,1])
axr.set_xlim(0,1)
axr.set_ylim(len(t)-.5,-.5)
axr.axis('off')

# Brackets are clipped to the exact first/last row edges and never alter ax limits.
for name,pos,n in bounds:
    y=pos+(n-1)/2
    y0=max(-.5,pos-.5)
    y1=min(len(t)-.5,pos+n-.5)
    axr.plot([.02,.02],[y0,y1],lw=.65,alpha=.55,clip_on=True)
    axr.plot([.02,.08],[y0,y0],lw=.65,alpha=.55,clip_on=True)
    axr.plot([.02,.08],[y1,y1],lw=.65,alpha=.55,clip_on=True)
    ytext=y
    if name=='late_recruitment':
        ytext=y-1.0
    elif name=='progressive_increase':
        ytext=y+1.0
    axr.text(.13,ytext,f'{pretty[name]}\n$n={n}$',va='center',ha='left',fontsize=6.5)

# Hard invariant: the image axis must remain exactly 96 contiguous raster rows.
ax.set_ylim(len(t)-.5,-.5)
cax=fig.add_axes([.19,.055,.42,.022])
cb=fig.colorbar(im,cax=cax,orientation='horizontal')
cb.set_label('Mean endpoint-conditioned relational distortion',fontsize=6.5)
cb.ax.tick_params(labelsize=6,length=2)
fig.subplots_adjust(left=.11,right=.96,top=.98,bottom=.23)
save(fig,O5,'panel_D')

# E — 25-state molecular landscape with objectively selected named trajectories
MOL=RES/'Fig5_Disease_Trajectories'/'gw_biology_dense_v5'/'molecular'/'gene_integrated_scale_associations.csv'
g=pd.read_csv(MOL)
need={'u','gene','cell_spearman','object_spearman_sqrt_mass',
      'three_way_score','three_way_direction_concordant'}
if not need.issubset(g.columns):
    raise RuntimeError(f'Fig5E missing {sorted(need-set(g.columns))}')
for c in ['u','cell_spearman','object_spearman_sqrt_mass','three_way_score']:
    g[c]=pd.to_numeric(g[c],errors='coerce')
g=g.dropna(subset=['u','gene','cell_spearman']).copy()
us=np.sort(g.u.unique())
if len(us)!=25 or not np.allclose(us,np.arange(.08,.58,.02),atol=1e-8):
    raise RuntimeError(f'Fig5E expected measured u=0.08..0.56 by 0.02; found {us}')

# Reproduce the frozen objective ranking used to nominate trajectories.
rank=[]
for gene,d in g.groupby('gene'):
    d=d.sort_values('u')
    x=d.cell_spearman.to_numpy(float)
    y=d.object_spearman_sqrt_mass.to_numpy(float)
    u=d.u.to_numpy(float)
    rank.append({
        'gene':gene,'n_states':len(d),
        'max_abs_cell_assoc':np.nanmax(np.abs(x)),
        'max_abs_object_assoc':np.nanmax(np.abs(y)),
        'association_range':np.nanmax(x)-np.nanmin(x),
        'early_mean':np.nanmean(x[u<=.20]),
        'middle_mean':np.nanmean(x[(u>=.32)&(u<=.44)]),
        'late_mean':np.nanmean(x[u>=.46]),
        'median_three_way_score':np.nanmedian(d.three_way_score),
        'concordant_fraction':np.nanmean(d.three_way_direction_concordant.astype(float))
    })
rk=pd.DataFrame(rank)
for c in ['max_abs_cell_assoc','association_range','median_three_way_score','concordant_fraction']:
    rk[c+'_pct']=rk[c].rank(pct=True)
rk['rank_score']=sum(rk[c+'_pct'] for c in
    ['max_abs_cell_assoc','association_range','median_three_way_score','concordant_fraction'])
rk=rk.sort_values(['rank_score','gene'],ascending=[False,True]).reset_index(drop=True)

# Representatives are selected reproducibly from the ranking:
# six highest-ranked genes in the dominant negative→positive→negative trajectory
# plus two highest-ranked genes in the opposing positive→negative→positive trajectory.
eps=.005
dom=rk[(rk.early_mean < -eps)&(rk.middle_mean > eps)&(rk.late_mean < -eps)].head(6)
opp=rk[(rk.early_mean > eps)&(rk.middle_mean < -eps)&(rk.late_mean > eps)].head(2)
sel=pd.concat([dom,opp],ignore_index=True).drop_duplicates('gene')
if len(sel)!=8:
    raise RuntimeError(f'Fig5E expected 8 objective representatives, got {len(sel)}')
highlight=sel.gene.tolist()

# Full 359-gene field with objective highlighted trajectories.
fig,ax=plt.subplots(figsize=(5.45,3.45))
for gene,d in g.groupby('gene'):
    d=d.sort_values('u')
    ax.plot(d.u,d.cell_spearman,lw=.28,alpha=.085,zorder=1)

env=(g.groupby('u').cell_spearman
       .quantile([.10,.50,.90]).unstack().sort_index())
ax.fill_between(env.index.to_numpy(float),env[.10].to_numpy(float),
                env[.90].to_numpy(float),alpha=.07,zorder=0)
ax.axhline(0,lw=.6,alpha=.32,zorder=0)

# Draw the eight selected trajectories without labels over the data field.
curves={}
for gene in highlight:
    d=g[g.gene==gene].sort_values('u')
    curves[gene]=d
    ax.plot(d.u,d.cell_spearman,lw=1.05,alpha=.92,zorder=3)

# Conventional gene legend below the axis; no leaders through the data field.
from matplotlib.lines import Line2D
handles=[]
labels=[]
for gene in highlight:
    d=curves[gene]
    # Draw once, then construct a proxy legend handle from the actual line color.
    drawn=ax.plot(d.u,d.cell_spearman,lw=1.08,alpha=.94,zorder=4)[0]
    handles.append(Line2D([0],[0],color=drawn.get_color(),lw=1.35))
    labels.append(gene)

# Small bracket indicating the interval used in the objective trajectory-family classification.
y_top=ax.get_ylim()[1] if ax.get_ylim()[1] > .30 else .36
yb=y_top*0.965
tick=(ax.get_ylim()[1]-ax.get_ylim()[0])*.018
ax.plot([.32,.44],[yb,yb],lw=.65,alpha=.55,clip_on=False)
ax.plot([.32,.32],[yb-tick,yb],lw=.65,alpha=.55,clip_on=False)
ax.plot([.44,.44],[yb-tick,yb],lw=.65,alpha=.55,clip_on=False)
ax.text(.38,yb+tick*.18,'middle interval',ha='center',va='bottom',fontsize=5.8)

ax.set_xlabel('Hierarchy progression, $u$')
ax.set_ylabel('Association with PRCC relational deviation\n(cell-level Spearman $r$)')
ax.set_xlim(.08,.56)
ax.set_xticks(np.arange(.08,.57,.08))
clean(ax)
leg=ax.legend(handles,labels,frameon=False,ncol=4,fontsize=6.3,
              loc='upper center',bbox_to_anchor=(.5,-.23),
              handlelength=1.55,columnspacing=1.25,handletextpad=.45)
for txt in leg.get_texts():
    txt.set_fontstyle('italic')
fig.subplots_adjust(left=.15,right=.98,bottom=.34,top=.95)
save(fig,O5,'panel_E')

rk.to_csv(O5/'panel_E_gene_trajectory_ranking.csv',index=False)
sel.to_csv(O5/'panel_E_highlighted_genes.csv',index=False)

pd.DataFrame({'panel':list('ABCDE'),'status':['rendered','rendered','copied frozen V6','rendered','rendered'],'scientific_recalculation':[False]*5}).to_csv(O5/'audit.csv',index=False)
(O5/'manifest.json').write_text(json.dumps({'figure':'MainFig5','mode':'standalone_panels','assembled_figure':False,'scientific_recalculation':False,'panel_C':'frozen V6 exact copy','panel_E_note':'module-level trajectory uses previously audited frozen summary; exact 19-gene membership recovery not required for this panel'},indent=2))

# ---------------- Fig 6 ----------------
SRC=RES/'Fig6_Consolidation_Prediction'/'restricted_dense_v4'
PERF=pd.read_csv(SRC/'performance_all.csv'); MEAN=pd.read_csv(SRC/'performance_means.csv'); IMP=pd.read_csv(SRC/'permutation_importance.csv')
SAMPLES=[('healthy_reference','Healthy brain'),('nondiseased_kidney','Nondiseased kidney')]
MAP={s:pd.read_parquet(SRC/f'{s}_oof_predictions.parquet') for s,_ in SAMPLES}

# Fail-closed schema + numerical preflight.
def require_cols(df, cols, label):
    missing=[c for c in cols if c not in df.columns]
    if missing: raise RuntimeError(f'{label}: missing required columns {missing}; available={list(df.columns)}')
require_cols(PERF,['sample','analysis','task','information_class','auc','mae_improvement'],'Fig6 performance_all.csv')
require_cols(MEAN,['sample','analysis','task','horizon','auc','mae_improvement'],'Fig6 performance_means.csv')
require_cols(IMP,['sample','task','feature','importance'],'Fig6 permutation_importance.csv')
for sample,_ in SAMPLES:
    require_cols(MAP[sample],['x_plot','y_plot','observed_persistent','predicted_persistent_probability','observed_consolidation_u','predicted_consolidation_u'],f'Fig6 {sample}_oof_predictions.parquet')

classes=['geometry','geometry_topology','geometry_topology_mechanics']
for sample,_ in SAMPLES:
    for task,metric in [('persistence','auc'),('consolidation_scale','mae_improvement')]:
        q=PERF[(PERF['sample']==sample)&(PERF['analysis']=='information_class')&(PERF['task']==task)].copy()
        absent=[c for c in classes if c not in set(q['information_class'].dropna())]
        if absent: raise RuntimeError(f'Fig6 information classes missing for {sample}/{task}: {absent}')
        for c in classes:
            z=pd.to_numeric(q.loc[q.information_class==c,metric],errors='coerce').dropna()
            if len(z)<2: raise RuntimeError(f'Fig6 {sample}/{task}/{c}: need >=2 finite fold values, got {len(z)}')
        qh=MEAN[(MEAN['sample']==sample)&(MEAN['analysis']=='horizon')&(MEAN['task']==task)].copy()
        hv=pd.to_numeric(qh['horizon'],errors='coerce')
        mv=pd.to_numeric(qh[metric],errors='coerce')
        ok=np.isfinite(hv)&np.isfinite(mv)
        if ok.sum()<3 or len(np.unique(hv[ok]))<3:
            raise RuntimeError(f'Fig6 {sample}/{task}: horizon trajectory is empty/degenerate')
        qi=IMP[(IMP['sample']==sample)&(IMP['task']==task)].copy()
        iv=pd.to_numeric(qi['importance'],errors='coerce')
        if np.isfinite(iv).sum()<3 or float(np.nanmax(np.abs(iv)))<=1e-12:
            raise RuntimeError(f'Fig6 {sample}/{task}: permutation importance empty/zero; refusing to render')
print('PASS — Fig6 source/schema/numerical preflight')

# Write the exact selected numeric sources before plotting.
PERF.to_csv(O6/'source_performance_all_selected.csv',index=False)
MEAN.to_csv(O6/'source_performance_means_selected.csv',index=False)
IMP.to_csv(O6/'source_permutation_importance_selected.csv',index=False)

def spatial(ax,df,col,cmap,vmin=None,vmax=None,s=1.35):
    x=pd.to_numeric(df.x_plot,errors='coerce').to_numpy()
    y=pd.to_numeric(df.y_plot,errors='coerce').to_numpy()
    z=pd.to_numeric(df[col],errors='coerce').to_numpy()
    ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(z)
    if ok.sum()<100: raise RuntimeError(f'{col}: too few finite spatial values ({ok.sum()})')
    sc=ax.scatter(x[ok],y[ok],c=z[ok],s=s,cmap=cmap,vmin=vmin,vmax=vmax,linewidths=0,rasterized=True)
    xmin,xmax=np.nanmin(x[ok]),np.nanmax(x[ok]); ymin,ymax=np.nanmin(y[ok]),np.nanmax(y[ok])
    cx=(xmin+xmax)/2; cy=(ymin+ymax)/2; side=max(xmax-xmin,ymax-ymin)*1.015
    ax.set_xlim(cx-side/2,cx+side/2); ax.set_ylim(cy-side/2,cy+side/2)
    ax.set_aspect('equal'); ax.axis('off')
    return sc

# A/B — matched observed/OOF-predicted maps with dedicated colorbar columns
def spatial_panel(sample,stem):
    df=MAP[sample]
    fig=plt.figure(figsize=(4.75,4.45))
    gs=fig.add_gridspec(2,3,width_ratios=[1,1,.055],wspace=.035,hspace=.08)
    axs=np.array([[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])],
                  [fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]])
    cax0=fig.add_subplot(gs[0,2]); cax1=fig.add_subplot(gs[1,2])
    spatial(axs[0,0],df,'observed_persistent','coolwarm',0,1,s=1.65)
    sp=spatial(axs[0,1],df,'predicted_persistent_probability','coolwarm',0,1,s=1.65)
    uo=pd.to_numeric(df.observed_consolidation_u,errors='coerce').to_numpy()
    ok=np.isfinite(uo); lo=float(np.nanquantile(uo[ok],.01)); hi=float(np.nanquantile(uo[ok],.99))
    spatial(axs[1,0],df,'observed_consolidation_u','viridis',lo,hi,s=1.65)
    sc=spatial(axs[1,1],df,'predicted_consolidation_u','viridis',lo,hi,s=1.65)
    titles=[['Observed persistence','OOF predicted persistence'],
            ['Observed consolidation','OOF predicted consolidation']]
    for i in range(2):
        for j in range(2): axs[i,j].set_title(titles[i][j],fontsize=7.5,pad=1.5)
    cb0=fig.colorbar(sp,cax=cax0); cb0.set_label('Persistence probability',fontsize=6.8)
    cb1=fig.colorbar(sc,cax=cax1); cb1.set_label('Consolidation coordinate, $u_c$',fontsize=6.8)
    for cb in [cb0,cb1]: cb.ax.tick_params(labelsize=6,width=.5,length=2)
    fig.subplots_adjust(left=.015,right=.94,bottom=.015,top=.965)
    save(fig,O6,stem)
spatial_panel('healthy_reference','panel_A')
spatial_panel('nondiseased_kidney','panel_B')

# C — actual unconnected spatial-CV fold points, with mean bars.
labels=['Geometry','+ topology','+ mechanics']
fig,axs=plt.subplots(1,2,figsize=(6.15,2.65))
for si,(sample,label) in enumerate(SAMPLES):
    marker='o' if si==0 else 's'
    xoff=-.07 if si==0 else .07
    q=PERF[(PERF['sample']==sample)&(PERF['analysis']=='information_class')&(PERF['task']=='persistence')]
    for i,c in enumerate(classes):
        z=pd.to_numeric(q.loc[q.information_class==c,'auc'],errors='coerce').dropna().to_numpy()
        jitter=np.linspace(-.025,.025,len(z))
        axs[0].scatter(i+xoff+jitter,z,s=18,alpha=.75,marker=marker,label=label if i==0 else None)
        axs[0].plot([i+xoff-.055,i+xoff+.055],[z.mean(),z.mean()],lw=1.6)
    q=PERF[(PERF['sample']==sample)&(PERF['analysis']=='information_class')&(PERF['task']=='consolidation_scale')]
    for i,c in enumerate(classes):
        z=100*pd.to_numeric(q.loc[q.information_class==c,'mae_improvement'],errors='coerce').dropna().to_numpy()
        jitter=np.linspace(-.025,.025,len(z))
        axs[1].scatter(i+xoff+jitter,z,s=18,alpha=.75,marker=marker,label=label if i==0 else None)
        axs[1].plot([i+xoff-.055,i+xoff+.055],[z.mean(),z.mean()],lw=1.6)
axs[0].axhline(.5,ls='--',lw=.65,alpha=.4); axs[1].axhline(0,ls='--',lw=.65,alpha=.4)
axs[0].set_ylabel('Held-out persistence ROC AUC')
axs[1].set_ylabel('Held-out MAE improvement (%)')
for ax in axs:
    ax.set_xticks(range(3),labels)
    ax.tick_params(axis='x',rotation=12,labelsize=6.5)
    clean(ax)
# minimalist tissue key without connecting fold points
axs[0].legend(frameon=False,loc='best',fontsize=6.2)
fig.tight_layout(pad=.45,w_pad=1.0)
save(fig,O6,'panel_C')

# D — connected dense horizon trajectories; connection is legitimate because h is ordered.
fig,axs=plt.subplots(1,2,figsize=(6.15,2.65))
for sample,label in SAMPLES:
    q=MEAN[(MEAN['sample']==sample)&(MEAN['analysis']=='horizon')&(MEAN['task']=='persistence')].copy()
    q=q.assign(horizon=pd.to_numeric(q.horizon,errors='coerce'),auc=pd.to_numeric(q.auc,errors='coerce')).dropna(subset=['horizon','auc']).sort_values('horizon')
    axs[0].plot(q.horizon,q.auc,'o-',lw=1.35,ms=3.2,label=label)
    q=MEAN[(MEAN['sample']==sample)&(MEAN['analysis']=='horizon')&(MEAN['task']=='consolidation_scale')].copy()
    q=q.assign(horizon=pd.to_numeric(q.horizon,errors='coerce'),mae_improvement=pd.to_numeric(q.mae_improvement,errors='coerce')).dropna(subset=['horizon','mae_improvement']).sort_values('horizon')
    axs[1].plot(q.horizon,100*q.mae_improvement,'o-',lw=1.35,ms=3.2,label=label)
axs[0].axhline(.5,ls='--',lw=.65,alpha=.4); axs[1].axhline(0,ls='--',lw=.65,alpha=.4)
axs[0].set(xlabel='Hierarchy horizon, $h$',ylabel='Persistence ROC AUC')
axs[1].set(xlabel='Hierarchy horizon, $h$',ylabel='MAE improvement (%)')
for ax in axs: clean(ax)
axs[1].legend(frameon=False,loc='best')
fig.tight_layout(pad=.45,w_pad=1.0)
save(fig,O6,'panel_D')

# E — calibration, compact and equal-aspect where possible.
fig,axs=plt.subplots(1,2,figsize=(6.5,2.9))
for sample,label in SAMPLES:
    df=MAP[sample]
    p=pd.to_numeric(df.predicted_persistent_probability,errors='coerce').to_numpy()
    y=pd.to_numeric(df.observed_persistent,errors='coerce').to_numpy()
    ok=np.isfinite(p)&np.isfinite(y); p=p[ok]; y=y[ok]
    bins=np.unique(np.quantile(p,np.linspace(0,1,9))); bx=[];by=[]
    for lo0,hi0 in zip(bins[:-1],bins[1:]):
        m=(p>=lo0)&(p<(hi0) if hi0<bins[-1] else p<=hi0)
        if m.sum()>=20: bx.append(p[m].mean()); by.append(y[m].mean())
    axs[0].plot(bx,by,'o-',lw=1.2,ms=3.2,label=label)
    p=pd.to_numeric(df.predicted_consolidation_u,errors='coerce').to_numpy()
    y=pd.to_numeric(df.observed_consolidation_u,errors='coerce').to_numpy()
    ok=np.isfinite(p)&np.isfinite(y); p=p[ok]; y=y[ok]
    bins=np.unique(np.quantile(p,np.linspace(0,1,9))); bx=[];by=[]
    for lo0,hi0 in zip(bins[:-1],bins[1:]):
        m=(p>=lo0)&(p<(hi0) if hi0<bins[-1] else p<=hi0)
        if m.sum()>=20: bx.append(p[m].mean()); by.append(y[m].mean())
    axs[1].plot(bx,by,'o-',lw=1.2,ms=3.2,label=label)
axs[0].plot([0,1],[0,1],'--',lw=.65,alpha=.35)
axs[0].set(xlabel='Predicted persistence probability',ylabel='Observed persistent fraction',xlim=(0,1),ylim=(0,1))
axs[0].set_aspect('equal',adjustable='box')
all_u=np.concatenate([pd.to_numeric(MAP[s].observed_consolidation_u,errors='coerce').to_numpy() for s,_ in SAMPLES])
all_u=all_u[np.isfinite(all_u)]; lo=float(all_u.min()); hi=float(all_u.max())
axs[1].plot([lo,hi],[lo,hi],'--',lw=.65,alpha=.35)
axs[1].set(xlabel='Predicted consolidation coordinate',ylabel='Observed consolidation coordinate',xlim=(lo,hi),ylim=(lo,hi))
axs[1].set_aspect('equal',adjustable='box')
for ax in axs: clean(ax)
axs[0].legend(frameon=False,loc='lower right')
fig.tight_layout(pad=.55,w_pad=1.2)
save(fig,O6,'panel_E')

# F — held-out permutation attribution from the actual nonzero source rows
friendly={
'delta_pressure_mean':'Pressure difference',
'mechanics_degree_abs_difference':'Mechanical-degree difference',
'interface_area':'Interface area',
'contact_area':'Interface area',
'length_asymmetry':'Length asymmetry',
'geometry_length_asymmetry':'Length asymmetry',
'centroid_distance':'Centroid distance',
'degree_abs_difference':'Topological-degree difference'
}
I=IMP.copy()
I['importance']=pd.to_numeric(I['importance'],errors='coerce')
I=I[np.isfinite(I.importance)].copy()
if I.empty or float(I.importance.abs().max())<=1e-12:
    raise RuntimeError('Fig6F: permutation table is empty/zero')

# Respect exact task names in source; identify persistence and scale tasks by content.
tasks=list(map(str,I['task'].dropna().unique()))
ptask=next((x for x in tasks if 'persist' in x.lower()),None)
stask=next((x for x in tasks if ('scale' in x.lower() or 'consolid' in x.lower()) and x!=ptask),None)
if ptask is None or stask is None:
    raise RuntimeError(f'Fig6F: could not identify persistence/scale tasks from {tasks}')

rank=(I.groupby('feature')['importance'].apply(lambda x: float(np.nanmax(np.abs(x))))
      .sort_values(ascending=False))
features=rank.head(7).index.tolist()
fig,axs=plt.subplots(1,2,figsize=(7.0,3.45),sharey=True)
for ax,task,xlab in zip(axs,[ptask,stask],
                        ['Held-out AUC loss','Held-out MAE increase']):
    for yi,f in enumerate(features):
        for si,(sample,label) in enumerate(SAMPLES):
            vals=I[(I['sample']==sample)&(I['task']==task)&(I['feature']==f)]['importance'].to_numpy(float)
            if len(vals)==0: continue
            off=(-.10 if si==0 else .10)
            yy=np.full(len(vals),yi+off)
            ax.scatter(vals,yy,s=15,alpha=.65,marker=('o' if si==0 else 's'),
                       label=label if yi==0 else None)
            if len(vals)>1:
                ax.plot([np.nanmin(vals),np.nanmax(vals)],[yi+off,yi+off],lw=.55,alpha=.45,zorder=0)
            ax.plot([np.nanmean(vals),np.nanmean(vals)],[yi+off-.055,yi+off+.055],lw=1.45)
    ax.axvline(0,ls='--',lw=.6,alpha=.35)
    ax.set_xlabel(xlab); clean(ax)
axs[0].set_yticks(range(len(features)),[friendly.get(f,f.replace('_',' ')) for f in features],fontsize=6.5)
axs[0].invert_yaxis()
axs[0].legend(frameon=False,fontsize=6.2,loc='lower right')
fig.tight_layout(pad=.45,w_pad=.75)
save(fig,O6,'panel_F')

# Post-render data assertions for the three panels that previously failed silently.
for sample,_ in SAMPLES:
    q=PERF[(PERF['sample']==sample)&(PERF['analysis']=='information_class')]
    if q[['auc','mae_improvement']].notna().sum().sum()==0:
        raise RuntimeError(f'Fig6C post-render assertion failed for {sample}')
    qh=MEAN[(MEAN['sample']==sample)&(MEAN['analysis']=='horizon')]
    if qh[['auc','mae_improvement']].notna().sum().sum()==0:
        raise RuntimeError(f'Fig6D post-render assertion failed for {sample}')
    qi=IMP[IMP['sample']==sample]
    if len(qi)==0 or float(pd.to_numeric(qi.importance,errors='coerce').abs().max())<=1e-12:
        raise RuntimeError(f'Fig6F post-render assertion failed for {sample}')
print('PASS — Fig6 C/D/F nonblank rendering assertions')

pd.DataFrame({'panel':list('ABCDEF'),'status':['rendered']*6,'scientific_recalculation':[False]*6}).to_csv(O6/'audit.csv',index=False)
(O6/'manifest.json').write_text(json.dumps({'figure':'MainFig6','source':str(SRC),'mode':'standalone_panels','assembled_figure':False,'scientific_recalculation':False,'spatial_maps':'out-of-fold predictions only','hierarchy_updated_predictors_used':False,'biological_time_claim':False},indent=2))
print('PASS — publication-layout rebuild rendered MainFig5 A–E and MainFig6 A–F')
print('OUTPUT:',OUTROOT)
