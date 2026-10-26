"""Execute two frozen continued-pretraining capacity arms sequentially on one GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,state):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan_path=Path(a.plan);digest=sha(plan_path);plan=json.loads(plan_path.read_text())
    out=ROOT/plan['output'];state_path=out/'state.json'
    if state_path.exists():raise ValueError('Refuse to restart or overwrite a series')
    def verify():
        if sha(plan_path)!=digest:raise ValueError('Series plan changed')
        for name,expected in plan['sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Frozen input changed: '+name)
    verify();out.mkdir(parents=True,exist_ok=True)
    state={'schema':1,'status':'running','pid':os.getpid(),'plan_sha256':digest,'completed':[]}
    write(state_path,state)
    try:
        for arm in plan['arms']:
            verify();recipe=json.loads((ROOT/arm['plan']).read_text());cfg=recipe['training']
            command=[sys.executable,str(ROOT/'scripts/train_policy_value.py'),
                '--data',*recipe['data'],'--val',recipe['selection_validation'],
                '--initial',recipe['initial'],'--out',str(out/arm['id']),
                '--channels',str(cfg['channels']),'--blocks',str(cfg['residual_blocks']),
                '--epochs',str(cfg['epochs']),'--batch',str(cfg['batch']),
                '--lr',str(cfg['lr']),'--seed',str(cfg['seed']),'--plan',arm['plan']]
            started=time.monotonic()
            with (out/(arm['id']+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state.update(arm=arm['id'],child_pid=child.pid);write(state_path,state)
                code=child.wait()
            if code:raise RuntimeError(arm['id']+' training returned '+str(code))
            manifest=json.loads((out/arm['id']/'manifest.json').read_text())
            if manifest['status']!='completed':raise ValueError('Training manifest is not completed')
            verify()
            state['completed'].append({'id':arm['id'],'seconds':time.monotonic()-started,
                'manifest_sha256':sha(out/arm['id']/'manifest.json'),
                'model_sha256':sha(out/arm['id']/'best.npz')})
            write(state_path,state)
        state['status']='completed';write(state_path,state)
    except BaseException as error:
        state.update(status='failed',error=str(error));write(state_path,state);raise


if __name__=='__main__':main()
