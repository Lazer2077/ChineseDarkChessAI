"""Serialize a new fixed experiment after a verified live dependency finishes."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());digest=sha(a.plan);dependency=plan['wait_for']
    marker=ROOT/(plan['out']+'.wait.json')
    if marker.exists():raise ValueError('Refuse to overwrite/restart a queued experiment')
    def save(status,**extra):
        marker.write_text(json.dumps({'schema':1,'pid':os.getpid(),'status':status,
            'plan_sha256':digest,**extra},indent=2)+'\n')
    try:
        while True:
            if sha(a.plan)!=digest:raise ValueError('Queued plan changed')
            for name,h in plan['sha256'].items():
                if sha(ROOT/name)!=h:raise ValueError('Frozen input changed: '+name)
            state=json.loads((ROOT/dependency['state']).read_text())
            if state['plan_sha256']!=dependency['plan_sha256']:raise ValueError('Wrong dependency recipe')
            if state['status']=='completed':break
            if state['status']!='running':raise ValueError('Dependency did not complete cleanly')
            command=Path('/proc')/str(state['pid'])/'cmdline'
            if not command.exists() or b'scripts/run_frozen_experiment.py' not in command.read_bytes():
                raise ValueError('Dependency controller missing; never restart it here')
            save('waiting',dependency_pid=state['pid']);time.sleep(15)
        save('dispatched')
        os.execv(sys.executable,[sys.executable,str(ROOT/'scripts/run_frozen_experiment.py'),'--plan',a.plan])
    except BaseException as error:save('failed',error=str(error));raise


if __name__=='__main__':main()
