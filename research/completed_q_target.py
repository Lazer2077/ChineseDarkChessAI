"""Completed-Q policy target from Danihelka et al.(ICLR2022), Eq11/33.

Pure vector transformation on legal root actions. It does not implement the
paper's action sampling/interior search or prove improvement with approximate
information-set action values.
"""
import math
import numpy as np


def completed_q_target(prior,visits,root_q,root_value,scale):
    p=np.asarray(prior,dtype=np.float64);n=np.asarray(visits);q=np.asarray(root_q,dtype=np.float64)
    if p.ndim!=1 or not len(p) or n.shape!=p.shape or q.shape!=p.shape:
        raise ValueError('Require matching nonempty legal-action vectors')
    if not np.issubdtype(n.dtype,np.integer) or (n<0).any():raise ValueError('Invalid completed visit counts')
    if not np.isfinite(p).all() or (p<0).any() or not np.isfinite(p.sum()) or p.sum()<=0:
        raise ValueError('Invalid legal priors')
    if not math.isfinite(root_value) or not math.isfinite(scale) or scale<0:raise ValueError('Invalid root value/scale')
    seen=n>0
    if not np.isfinite(q[seen]).all():raise ValueError('Nonfinite visited action value')
    p=p/p.sum();total=sum(map(int,n))
    if total:
        mass=float(p[seen].sum())
        if mass<=0:raise ValueError('Visited prior mass is zero; mixed completion undefined')
        weighted_q=float(np.dot(p[seen],q[seen])/mass)
        mixed=(float(root_value)+total*weighted_q)/(total+1)
    else:mixed=float(root_value)
    completed=np.where(seen,q,mixed)
    if not np.isfinite(completed).all():raise ValueError('Invalid completed action values')
    if not scale or not total:target=p.copy()
    else:
        logits=np.full(len(p),-np.inf);support=p>0
        logits[support]=np.log(p[support])+scale*completed[support]
        logits-=np.max(logits);target=np.exp(logits);target/=target.sum()
    positive=target>0
    kl=float(np.sum(target[positive]*(np.log(target[positive])-np.log(p[positive]))))
    gain=float(np.dot(target-p,completed))
    if gain < -1e-10:raise ValueError('Exponential tilt decreased its own completed-Q objective')
    return {'policy':target,'completed_q':completed,'mixed_value':mixed,
            'estimated_completed_q_gain':gain,'target_to_prior_kl':kl}
