#!/usr/bin/env python3
"""Independent SUTRA SuppFig7 frozen-table renderer; not original analysis producer."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FILES=['content_distance_matrix','content_trajectory_long','disease_form_content','event_ledger_manifest','family_permutation_test','pairwise_distances','specimen_summary','structural_distance_matrix','structural_trajectory_long']
ORDER=['healthy_reference','alzheimers','gbm_reference_addon','nondiseased_kidney','prcc']
LABELS={'healthy_reference':'Healthy brain','alzheimers':"Alzheimer's",'gbm_reference_addon':'GBM-related reference','nondiseased_kidney':'Nondiseased kidney','prcc':'PRCC'}

def save(fig,out,name):
    fig.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(out/f'{name}.{ext}',bbox_inches='tight',**({'dpi':300} if ext=='png' else {}))
    plt.close(fig)

def check(t):
    for typ in ['structural','content']:
        d=t[f'{typ}_trajectory_long']
        assert {'sample','u','organ'}.issubset(d.columns)
        assert set(d['sample'])==set(ORDER)
        assert d['u'].between(0,1).all()
        for _,g in d.groupby('sample'):
            assert g.u.is_monotonic_increasing and not g.u.duplicated().any()
        m=t[f'{typ}_distance_matrix'].set_index('sample').loc[ORDER,ORDER].to_numpy(float)
        assert np.allclose(m,m.T) and np.allclose(np.diag(m),0)
    p=t['family_permutation_test']
    assert set(p.analysis)=={'structural_form_alignment','biological_content_alignment'}
    assert (p.n_permutations==5000).all()
    d=t['disease_form_content']
    assert set(zip(d.reference,d.disease))=={('healthy_reference','alzheimers'),('nondiseased_kidney','prcc')}
    for _,r in t['pairwise_distances'].iterrows():
        typ='structural' if r['kind']=='structural_form' else 'content'
        m=t[f'{typ}_distance_matrix'].set_index('sample')
        assert np.isclose(m.loc[r.sample_a,r.sample_b],r.distance,rtol=1e-8,atol=1e-8)

def panel_a(t,out):
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for col,typ in enumerate(['structural','content']):
        d=t[f'{typ}_trajectory_long']
        fields=[c for c in d if c.endswith('__z')]
        if not fields: raise ValueError(f'Missing precomputed standardized columns for {typ}')
        for row,organ in enumerate(['brain','kidney']):
            ax=axs[row,col]
            for s in ORDER:
                g=d[(d['sample']==s)&(d['organ']==organ)].sort_values('u')
                if len(g):
                    ax.plot(g.u,g[fields].mean(axis=1,skipna=True),label=LABELS[s],lw=1.7)
            ax.set(title=f'{organ.capitalize()} · {typ}',xlabel='Native organizational coordinate u',ylabel='Mean frozen standardized channel')
            ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False)
    fig.suptitle('A · Organizational trajectories (descriptive summary)')
    save(fig,out,'SuppFig7_A_frozen_trajectories')

def panel_b(t,out):
    fig,axs=plt.subplots(1,2,figsize=(13,6))
    for ax,typ in zip(axs,['structural','content']):
        m=t[f'{typ}_distance_matrix'].set_index('sample').loc[ORDER,ORDER].to_numpy(float)
        im=ax.imshow(m,cmap='viridis')
        ax.set_xticks(range(5),[LABELS[s] for s in ORDER],rotation=45,ha='right',fontsize=8)
        ax.set_yticks(range(5),[LABELS[s] for s in ORDER],fontsize=8)
        for i in range(5):
            for j in range(5): ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if m[i,j]>(m.max()+m.min())/2 else 'black')
        ax.set_title(f'{typ.capitalize()} distances');fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('B · Frozen organizational distance matrices')
    save(fig,out,'SuppFig7_B_frozen_distances')

def panel_c(t,out):
    d=t['family_permutation_test'].set_index('analysis').loc[['structural_form_alignment','biological_content_alignment']]
    fig,ax=plt.subplots(figsize=(8,5));x=np.arange(2)
    med=d.null_median.to_numpy(float)
    ax.errorbar(x+.12,med,yerr=[med-d.null_q025.to_numpy(float),d.null_q975.to_numpy(float)-med],fmt='o',capsize=6,label='Null median and 95% interval')
    ax.scatter(x-.12,d.observed_dispersion.to_numpy(float),marker='D',s=80,label='Observed dispersion')
    for i,(_,r) in enumerate(d.iterrows()): ax.annotate(f'p = {r.p_alignment:.6f}',(i-.12,r.observed_dispersion),xytext=(0,-20),textcoords='offset points',ha='center')
    ax.set_xticks(x,['Structural form','Biological content']);ax.set_ylabel('Dispersion')
    ax.set_title('C · Frozen circular-shift null summaries (5,000 permutations)')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    save(fig,out,'SuppFig7_C_frozen_alignment')

def panel_d(t,out):
    d=t['disease_form_content'];x=np.arange(len(d));fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(x-.18,d.structural_form_distance,.36,label='Structural form')
    ax.bar(x+.18,d.biological_content_distance,.36,label='Biological content')
    ax.set_xticks(x,["Brain: healthy–Alzheimer's" if r.organ=='brain' else 'Kidney: nondiseased–PRCC' for _,r in d.iterrows()])
    ax.set(ylabel='Frozen pairwise distance',title='D · Healthy–disease form and content')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    save(fig,out,'SuppFig7_D_frozen_form_content')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-data',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    t={name:pd.read_csv(a.source_data/f'{name}.csv') for name in FILES}
    check(t)
    for f in [panel_a,panel_b,panel_c,panel_d]:f(t,a.out)
    (a.out/'VALIDATION.json').write_text(json.dumps({'status':'PASS','scope':'Frozen-table consistency and alternative plotting only; upstream analysis and original panel identity not certified','files':FILES},indent=2)+'\n')
    print('PASS: nine frozen tables validated; four independent alternative panels rendered.')
if __name__=='__main__':main()
