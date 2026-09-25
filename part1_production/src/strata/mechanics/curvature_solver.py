from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse
from scipy.optimize import minimize


@dataclass(frozen=True)
class CurvatureSolverConfig:
    lambda_young_laplace: float = 1.0
    mu_scale: float = 100.0
    mu_gauge: float = 100.0
    ridge_tau: float = 1e-8
    ridge_p: float = 1e-8
    maxiter: int = 750
    gtol: float = 1e-6


def solve_curvature_constrained(A,Y,m,n,init_tau,init_p,cfg=CurvatureSolverConfig()):
    x0=np.r_[np.maximum(np.asarray(init_tau,float),0.0),np.asarray(init_p,float)]
    mt=float(np.mean(x0[:m])) if m else 1.0
    if mt<=1e-12:
        x0[:m]=1.0
    else:
        x0[:m]/=mt
        x0[m:]/=mt
    if n:
        x0[m:]-=float(np.mean(x0[m:]))

    def fg(x):
        tau=x[:m];p=x[m:]
        Ax=A@x
        val=0.5*float(np.dot(Ax,Ax))
        grad=np.asarray(A.T@Ax,float)

        if Y.shape[0]:
            Yx=Y@x
            lam=cfg.lambda_young_laplace
            val+=0.5*lam*float(np.dot(Yx,Yx))
            grad+=lam*np.asarray(Y.T@Yx,float)

        ds=float(np.mean(tau))-1.0
        val+=0.5*cfg.mu_scale*ds*ds
        grad[:m]+=cfg.mu_scale*ds/max(m,1)

        if n:
            dg=float(np.mean(p))
            val+=0.5*cfg.mu_gauge*dg*dg
            grad[m:]+=cfg.mu_gauge*dg/max(n,1)

        val+=0.5*cfg.ridge_tau*float(np.dot(tau,tau))
        grad[:m]+=cfg.ridge_tau*tau
        if n:
            val+=0.5*cfg.ridge_p*float(np.dot(p,p))
            grad[m:]+=cfg.ridge_p*p

        return val,grad

    bounds=[(0,None)]*m+[(None,None)]*n
    t0=time.time()
    res=minimize(
        fg,x0,method="L-BFGS-B",jac=True,bounds=bounds,
        options={"maxiter":cfg.maxiter,"gtol":cfg.gtol,"ftol":1e-12,"maxls":40}
    )

    tau=res.x[:m].copy()
    p=res.x[m:].copy()
    mt=float(np.mean(tau))
    if mt<=1e-12:
        raise RuntimeError("vanishing tension scale")
    tau/=mt
    p/=mt
    p-=float(np.mean(p))

    x=np.r_[tau,p]
    vr=A@x
    vertex_resid=np.sqrt(vr[0::2]**2+vr[1::2]**2)
    yl_resid=Y@x if Y.shape[0] else np.array([],float)

    return {
        "tension":tau,
        "pressure":p,
        "vertex_residual":vertex_resid,
        "young_laplace_residual":np.asarray(yl_resid,float),
        "success":bool(res.success),
        "message":str(res.message),
        "iterations":int(res.nit),
        "optimality":float(np.linalg.norm(res.jac,np.inf)),
        "runtime_seconds":float(time.time()-t0),
    }
