"""Reuse the frozen RL trainer with one process-local normalization-mode switch."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
import train_neural_selfplay as trainer
from research.frozen_normalization import freeze_statistics,running_buffers
from research.neural_replay import sha,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());captured=[]
    original=trainer.load_model
    def load(directory,device):
        net,metadata=original(directory,device);before=running_buffers(net)
        if plan['freeze_normalization_statistics']:net=freeze_statistics(net)
        captured.append((net,before));return net,metadata
    trainer.load_model=load
    trainer.main()
    if len(captured)!=1:raise ValueError('Unexpected model-load count')
    net,before=captured[0];after=running_buffers(net)
    changed=[key for key in before if not torch.equal(before[key],after[key])]
    if plan['freeze_normalization_statistics'] and changed:raise ValueError('Frozen normalization invariant failed')
    out=ROOT/plan['out']
    save_json(out/'normalization-proof.json',{'schema':1,'status':'completed',
        'frozen':plan['freeze_normalization_statistics'],'running_buffers':len(before),
        'changed_buffers':changed,'affine_parameters_remain_trainable':all(
            parameter.requires_grad for name,parameter in net.named_parameters()),
        'plan_sha256':sha(a.plan),'trainer_sha256':sha(ROOT/'scripts/train_neural_selfplay.py'),
        'scope':'Same fixed data/targets/optimizer/update recipe; only training normalization mode differs. No strength claim.'})


if __name__=='__main__':main()
