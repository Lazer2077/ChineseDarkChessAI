"""Legal-action teacher anchoring primitives; no training or actor replacement."""
import random
import numpy as np
import torch
from research.policy_value import AZ


def observer_legal_mask(info, visit_counts=None):
    """Reconstruct a mask from actor information, never the actual referee pool.

    This retains the pinned rules' full information-state history. It samples
    unknown removed inventory with a fixed internal seed; callers must disclose
    this observer-compatible mask profile and audit inventory sensitivity. A
    mismatch with stored search support is an error, never filtered silently.
    """
    world=info.determinize(random.Random(0))
    mask=np.zeros(AZ.MOVE_SIZE,dtype=bool)
    for move in world.legal_moves():mask[AZ.move_to_index(move)]=True
    if not mask.any():raise ValueError('Nonterminal policy record has no legal action')
    if visit_counts is not None:
        counts=np.asarray(visit_counts)
        if counts.shape!=(AZ.MOVE_SIZE,) or (counts<0).any() or not np.isfinite(counts).all():
            raise ValueError('Invalid stored visits')
        if counts[~mask].any():raise ValueError('Stored visit support differs from observer legal mask')
    return mask


def legal_teacher_kl(student_logits, teacher_logits, mask):
    """Mean KL(teacher_legal || student_legal), with no teacher gradient.

    The existing full-action search CE remains a separate objective. Invalid
    action mass is not preserved by this auxiliary loss. Avoid0*(-infinity)
    and retain exactly zero derivatives for masked actions.
    """
    if student_logits.ndim!=2 or student_logits.shape!=teacher_logits.shape or mask.shape!=student_logits.shape:
        raise ValueError('Require matching batch/action dimensions')
    if mask.dtype!=torch.bool or not bool(mask.any(1).all()):
        raise ValueError('Every policy record requires nonempty boolean legal support')
    if not bool(torch.isfinite(student_logits).all()) or not bool(torch.isfinite(teacher_logits).all()):
        raise ValueError('Nonfinite policy logits')
    student=student_logits.float();teacher=teacher_logits.detach().float()
    student_z=student.masked_fill(~mask,-torch.inf).logsumexp(1,keepdim=True)
    teacher_z=teacher.masked_fill(~mask,-torch.inf).logsumexp(1,keepdim=True)
    student_log=torch.where(mask,student-student_z,torch.zeros_like(student))
    teacher_log=torch.where(mask,teacher-teacher_z,torch.zeros_like(teacher))
    probability=torch.where(mask,teacher_log.exp(),torch.zeros_like(teacher))
    return (probability*(teacher_log-student_log)).sum(1).mean()
