"""Explicitly patched pinned RL trainer with an optional legal warm-policy KL.

The zero-coefficient path retains the original loss exactly. Original training
files and active experiments are never modified. This is an attributed auxiliary
distillation adaptation, not the full adaptive Kickstarting algorithm.
"""
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import train_neural_selfplay as original
from research.policy_anchor import legal_teacher_kl
from research.neural_replay import sha

ORIGINAL_SHA='0d8f6fdf9c5536ed8bf996a79a53e8d38b4224905a6f9a716b92fd1c61613e57'


def load_masks(plan,train_meta,raw,pi,device):
    folder=ROOT/plan['observer_masks'];report=original.json.loads((folder/'report.json').read_text())
    if report['status']!='completed' or sha(folder/'masks.npz')!=report['mask_arrays_sha256']:
        raise ValueError('Require completed frozen observer masks')
    expected=[(x['index'],x['sha256'],x['positions']) for x in train_meta]
    actual=[(x['index'],x['episode_npz_sha256'],x['positions']) for x in report['episodes']]
    if actual!=expected:raise ValueError('Mask episode order/digests differ from training data')
    with np.load(folder/'masks.npz',allow_pickle=False) as a:
        if set(a.files)!={'mask_bits','episode_indices','offsets'}:raise ValueError('Invalid mask arrays')
        if a['mask_bits'].dtype!=np.uint8 or a['mask_bits'].shape!=(len(raw),319):raise ValueError('Invalid packed masks')
        if list(a['episode_indices'])!=[x[0] for x in expected]:raise ValueError('Wrong mask game order')
        if list(a['offsets'])!=list(np.cumsum([0]+[x[2] for x in expected])):raise ValueError('Wrong mask offsets')
        mask=np.unpackbits(a['mask_bits'],axis=1,count=2550,bitorder=report['bitorder']).astype(bool)
    if not mask.any(1).all() or (pi[~mask]>0).any():raise ValueError('Invalid mask/search target support')
    return torch.from_numpy(mask).to(device)


def install():
    if sha(ROOT/'scripts/train_neural_selfplay.py')!=ORIGINAL_SHA:raise ValueError('Pinned original trainer changed')
    source=textwrap.dedent(inspect.getsource(original.main))
    def replace(needle,value):
        nonlocal source
        if source.count(needle)!=1:raise ValueError('Original training patch site changed: '+needle)
        source=source.replace(needle,value)
    replace("    optimizer=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=.0001)",
            """    anchor_lambda=float(plan['policy_anchor_lambda'])
    if not np.isfinite(anchor_lambda) or anchor_lambda<0:raise ValueError('Invalid anchor coefficient')
    masks=_load_masks(plan,train_meta,raw,pi,dev)
    teacher=None
    if anchor_lambda:
        teacher,_=load_model(ROOT/plan['initial_model'],dev)
        teacher.eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW(net.parameters(),lr=plan['lr'],weight_decay=.0001)""")
    replace("'scope':'Search-guided neural self-play RL update using actual root visits and terminal values. Tiny development pilot; no teacher-strength gain, equilibrium guarantee or model promotion.'",
            "'scope':'Search-visit/terminal-return update with optional legal-action warm-policy KL; no equilibrium, adaptive-Kickstarting or strength guarantee.', 'policy_anchor_lambda':anchor_lambda, 'observer_masks_sha256':sha(ROOT/plan['observer_masks']/'masks.npz')")
    replace("            net.train();order=torch.randperm(len(data),device=dev,generator=rng);sums=np.zeros(3)",
            "            net.train();order=torch.randperm(len(data),device=dev,generator=rng);sums=np.zeros(3);anchor_sum=0.")
    replace("                loss=ce+mse", """                anchor=logits.new_zeros(())
                if anchor_lambda:
                    legal=masks[ids]
                    legal=torch.where(flip[:,None],legal[:,mirror_actions],legal)
                    with torch.no_grad():teacher_logits,_=teacher(x)
                    anchor=_legal_teacher_kl(logits,teacher_logits,legal)
                    loss=ce+mse+anchor_lambda*anchor
                else:
                    loss=ce+mse
                anchor_sum+=float(anchor.detach())*len(ids)""")
    replace("'train_terminal_mse':float(sums[1]/sums[2]),'validation':metrics",
            "'train_terminal_mse':float(sums[1]/sums[2]),'train_legal_teacher_kl':anchor_sum/sums[2],'validation':metrics")
    namespace=dict(original.__dict__,_load_masks=load_masks,_legal_teacher_kl=legal_teacher_kl)
    exec(compile(source,'<cdca-legal-policy-anchor-training>','exec'),namespace)
    return namespace['main'],hashlib.sha256(source.encode()).hexdigest()


if __name__=='__main__':
    main,_=install();main()
