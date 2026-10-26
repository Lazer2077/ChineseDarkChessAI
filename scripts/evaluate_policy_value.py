"""Wait for both fixed training arms, then run all predeclared CPU development matches."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];plan=json.loads(Path(a.plan).read_text());out=ROOT/plan['out']
if out.exists():raise SystemExit('Use fresh evaluation output')
out.mkdir(parents=True);plan_hash=hashlib.sha256(Path(a.plan).read_bytes()).hexdigest()
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def state(stage,**extra):
    data={'stage':stage,'pid':os.getpid(),'plan_sha256':plan_hash,**extra}
    tmp=out/'state.json.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(out/'state.json')
def verify():
    if sha(a.plan)!=plan_hash:raise RuntimeError('Evaluation plan changed')
    for f,digest in plan['sha256'].items():
        if sha(ROOT/f)!=digest:raise RuntimeError('Frozen evaluation artifact changed: '+f)
try:
    state('waiting_for_fixed_training')
    while True:
        verify();completed=True
        for arm,directory in plan['models'].items():
            folder=ROOT/directory;marker=folder/'manifest.json'
            if not marker.exists():raise RuntimeError('Start evaluator after both training manifests exist')
            m=json.loads(marker.read_text())
            if m['status']=='failed':raise RuntimeError('A training arm failed; do not substitute another checkpoint')
            if m['status']!='completed':
                command=Path('/proc')/str(m['pid'])/'cmdline'
                if not command.exists() or 'scripts/train_policy_value.py' not in command.read_bytes().replace(b'\0',b' ').decode():
                    raise RuntimeError('Training process missing; never restart it from this evaluator')
                completed=False
        if completed:break
        time.sleep(5)
    model_hashes={}
    for arm,directory in plan['models'].items():
        folder=ROOT/directory;manifest=json.loads((folder/'manifest.json').read_text());best=json.loads((folder/'best.json').read_text())
        digest=sha(folder/'best.npz')
        if digest!=best['npz_sha256'] or digest!=manifest['best_npz_sha256']:raise RuntimeError('Selected model digest mismatch')
        if manifest['epochs_completed']!=plan['required_epochs'] or manifest['plan_sha256']!=plan['training_plan_sha256']:
            raise RuntimeError('Training recipe incomplete or different')
        model_hashes[arm]={'network':digest,'metadata':sha(folder/'best.json')}
    (out/'played-model-hashes.json').write_text(json.dumps(model_hashes,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JIEQI_NEURAL_PYTHON=sys.executable,
        YUAN_NATIVE_DIR=str(ROOT/'build/yuan-native/source'),JIEQI_RULES_BACKEND='cython',JIEQI_LEGAL_BACKEND='cython')
    for match in plan['matches']:
        verify()
        for arm,directory in plan['models'].items():
            if sha(ROOT/directory/'best.npz')!=model_hashes[arm]['network'] or sha(ROOT/directory/'best.json')!=model_hashes[arm]['metadata']:
                raise RuntimeError('Selected model changed between matches')
        state('playing',match=match['id'])
        candidate=match['candidate'];baseline=match['baseline']
        net=plan['teacher_net'] if baseline=='teacher' else str(Path(plan['models'][baseline])/'best.npz')
        binary=plan['teacher_binary'] if baseline=='teacher' else 'adapters/neural_policy_uci.sh'
        cmd=[sys.executable,'scripts/native_match.py','--candidate=adapters/neural_policy_uci.sh','--candidate-net='+str(Path(plan['models'][candidate])/'best.npz'),'--baseline='+binary,'--net='+net,'--tc='+plan['clock'],'--games='+str(match['games']),'--seed='+str(match['seed']),'--stage=development','--out='+str(out/match['id'])]
        with (out/(match['id']+'.log')).open('x') as log:subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    state('completed',matches=[m['id'] for m in plan['matches']])
except BaseException as error:
    state('failed',error=str(error));raise
