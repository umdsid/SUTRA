from dataclasses import dataclass
import numpy as np
from scipy import ndimage

@dataclass(frozen=True)
class RepairConfig:
    observed_secondary_max_pixels:int=10

ST=np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)

def audit_and_repair(labels, observed, cfg=RepairConfig()):
    labels=np.asarray(labels,dtype=np.int32)
    observed=np.asarray(observed,dtype=np.int32)
    repaired=labels.copy()
    maxlab=int(labels.max())
    slices=ndimage.find_objects(labels,max_label=maxlab)
    removed=np.zeros(labels.shape,bool)
    disc=[]; inferred=[]; obsdisc=[]; blocked=[]

    for lab in range(1,maxlab+1):
        sl=slices[lab-1] if lab-1<len(slices) else None
        if sl is None: continue
        crop=(labels[sl]==lab)
        cc,n=ndimage.label(crop,structure=ST)
        if n<=1: continue
        disc.append(lab)
        obs=(observed[sl]==lab)
        core=np.unique(cc[obs]); core=core[core>0]

        if len(core)==1:
            bad=crop & (cc!=int(core[0]))
            if bad.any():
                inferred.append(lab)
                rr=removed[sl]; rr[bad]=True; removed[sl]=rr
        elif len(core)>1:
            obsdisc.append(lab)
            sizes=np.bincount(cc[obs].ravel(),minlength=n+1)
            primary=int(core[np.argmax(sizes[core])])
            secondary=[int(k) for k in core if int(k)!=primary]
            too={k:int(sizes[k]) for k in secondary if int(sizes[k])>cfg.observed_secondary_max_pixels}
            if too:
                blocked.append({"label":lab,"secondary_observed_component_pixels":too})
                continue
            bad=np.isin(cc,secondary)
            rr=removed[sl]; rr[bad]=True; removed[sl]=rr

    if blocked:
        return labels.copy(), {
            "status":"HOLD",
            "completed_disconnected_labels":len(disc),
            "inferred_fragment_labels":len(inferred),
            "observed_disconnected_labels":len(obsdisc),
            "blocked_observed_labels":blocked,
            "removed_pixels":int(removed.sum())
        }

    # Removed disconnected fragments remain background.  A generic nearest-
    # foreground refill can assign the same source label back to the removed
    # island and recreate the exact disconnection we just repaired.  Leaving
    # these pixels unassigned is conservative: it removes inferred geometry
    # rather than fabricating a replacement label/contact.
    repaired[removed]=0

    post=[]
    slices=ndimage.find_objects(repaired,max_label=maxlab)
    for lab in range(1,maxlab+1):
        sl=slices[lab-1] if lab-1<len(slices) else None
        if sl is None: continue
        _,n=ndimage.label(repaired[sl]==lab,structure=ST)
        if n>1: post.append(lab)

    return repaired, {
        "status":"PASS" if not post else "HOLD",
        "completed_disconnected_labels":len(disc),
        "inferred_fragment_labels":len(inferred),
        "observed_disconnected_labels":len(obsdisc),
        "blocked_observed_labels":[],
        "removed_pixels":int(removed.sum()),
        "postrepair_disconnected_labels":len(post),
        "postrepair_examples":post[:20]
    }
