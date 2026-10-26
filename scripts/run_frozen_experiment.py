"""Run shell-free fixed experiment steps; never restart a prior output directory."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());digest=sha(a.plan);out=ROOT/plan['out']
    if out.exists():raise ValueError('Refuse to overwrite or restart an experiment')
    def verify():
        if sha(a.plan)!=digest:raise ValueError('Experiment plan changed')
        for name,expected in plan['sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Frozen input changed: '+name)
    verify();out.mkdir(parents=True)
    state={'schema':1,'status':'running','pid':os.getpid(),'plan_sha256':digest,'completed':[]}
    def save():
        tmp=out/'state.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'state.json')
    env=dict(os.environ,JIEQI_NEURAL_PYTHON=sys.executable,
        YUAN_NATIVE_DIR=str(ROOT/'build/yuan-native/source'),
        JIEQI_RULES_BACKEND='cython',JIEQI_LEGAL_BACKEND='cython',JIEQI_SEARCH_BACKEND='python')
    save()
    try:
        for step in plan['steps']:
            verify()
            with (out/(step['id']+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,*step['argv']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                state.update(stage=step['id'],child_pid=child.pid);save()
                if child.wait():raise RuntimeError('Experiment step failed: '+step['id'])
            verify();state['completed'].append(step['id']);state['child_pid']=None;save()
        state['status']='completed';save()
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
