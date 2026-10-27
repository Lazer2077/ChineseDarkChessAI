"""Per-record observable-entropy weights for a possible controlled KL ablation."""
from pathlib import Path
import json
import numpy as np
import torch
from research.policy_anchor import legal_teacher_kl
from research.observation_uncertainty import record_uncertainty
from research.neural_replay import sha

ROOT=Path(__file__).resolve().parents[1]


def weighted_legal_teacher_kl(student_logits,teacher_logits,mask,weights):
    if weights.ndim!=1 or len(weights)!=len(student_logits) or not weights.is_floating_point():
        raise ValueError('Require one floating coefficient per policy record')
    if not bool(torch.isfinite(weights).all()) or bool((weights<0).any()):
        raise ValueError('Invalid nonnegative finite coefficients')
    if student_logits.ndim!=2 or student_logits.shape!=teacher_logits.shape or mask.shape!=student_logits.shape:
        raise ValueError('Invalid policy dimensions')
    if mask.dtype!=torch.bool or not bool(mask.any(1).all()):raise ValueError('Empty or invalid legal mask')
    if not bool(torch.isfinite(student_logits).all()) or not bool(torch.isfinite(teacher_logits).all()):
        raise ValueError('Nonfinite logits')
    # Preserve the established constant-unit loss exactly, including reductions.
    if bool((weights==1).all()):return legal_teacher_kl(student_logits,teacher_logits,mask)
    student=student_logits.float();teacher=teacher_logits.detach().float()
    zs=student.masked_fill(~mask,-torch.inf).logsumexp(1,keepdim=True)
    zt=teacher.masked_fill(~mask,-torch.inf).logsumexp(1,keepdim=True)
    ls=torch.where(mask,student-zs,torch.zeros_like(student))
    lt=torch.where(mask,teacher-zt,torch.zeros_like(teacher))
    probability=torch.where(mask,lt.exp(),torch.zeros_like(teacher))
    per_record=(probability*(lt-ls)).sum(1)
    # Do not normalize by each minibatch's weight sum: the whole-corpus mean is1.
    return (weights.detach().to(per_record.dtype)*per_record).mean()


def load_entropy_weights(plan,train_meta,raw,device):
    mode=plan['entropy_weight_mode']
    if mode=='constant-unit':return torch.ones(len(raw),device=device)
    if mode!='mean-matched-observer-entropy':raise ValueError('Unknown entropy weighting mode')
    folder=ROOT/plan['uncertainty_data'];report=json.loads((folder/'report.json').read_text())
    if report['status']!='completed' or report['episodes']!=train_meta:
        raise ValueError('Entropy records differ from training episode order/hashes')
    if sha(folder/'diagnostic.npz')!=report['arrays_sha256']:raise ValueError('Entropy arrays changed')
    with np.load(folder/'diagnostic.npz',allow_pickle=False) as arrays:
        weights=arrays['candidate_mean_matched_weight'].copy()
    if weights.shape!=(len(raw),) or weights.dtype!=np.float64 or not np.isfinite(weights).all():
        raise ValueError('Invalid entropy coefficients')
    # Recompute using only the permitted record bytes, independent of the labels.
    uncertainty=record_uncertainty(raw)['normalized_entropy'];mean=float(uncertainty.mean())
    if mean<=0 or not np.allclose(weights,uncertainty/mean,rtol=0,atol=1e-12):
        raise ValueError('Coefficients do not match observable-only entropy')
    if not np.isclose(weights.mean(),1,rtol=0,atol=1e-12):raise ValueError('Global mean coefficient differs from1')
    return torch.from_numpy(weights.astype(np.float32)).to(device)
