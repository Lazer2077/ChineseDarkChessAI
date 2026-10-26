"""Wait for the frozen capacity series, then run fixed equal-clock diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());digest=sha(a.plan);out=ROOT/plan['out']
    if out.exists():raise ValueError('Refuse to overwrite an evaluation')
    def verify():
        if sha(a.plan)!=digest:raise ValueError('Evaluation plan changed')
        for name,expected in plan['sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Frozen input changed: '+name)
    verify();out.mkdir(parents=True)
    state={'schema':1,'pid':os.getpid(),'plan_sha256':digest,'stage':'waiting_for_training','matches':[]}
    def save():
        tmp=out/'state.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'state.json')
    def run(command,logfile):
        with logfile.open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save()
            if child.wait():raise RuntimeError('Evaluation child failed; see '+logfile.name)
        state['child_pid']=None
    env=dict(os.environ,JIEQI_NEURAL_PYTHON=sys.executable,
        YUAN_NATIVE_DIR=str(ROOT/'build/yuan-native/source'),
        JIEQI_RULES_BACKEND='cython',JIEQI_LEGAL_BACKEND='cython')
    save()
    try:
        while True:
            verify();training=json.loads((ROOT/plan['training_state']).read_text())
            if training['status']=='completed':break
            if training['status']=='failed':raise RuntimeError('Training failed; no replacement allowed')
            cmd=Path('/proc')/str(training['pid'])/'cmdline'
            if not cmd.exists() or b'scripts/residual_width_series.py' not in cmd.read_bytes():
                raise RuntimeError('Training controller missing; never restart it here')
            time.sleep(15)
        models={}
        for arm,directory in plan['models'].items():
            folder=ROOT/directory;manifest=json.loads((folder/'manifest.json').read_text());best=json.loads((folder/'best.json').read_text())
            if manifest['status']!='completed' or manifest['epochs_completed']!=2 or manifest['updates_planned']!=9442:
                raise ValueError('Incomplete training arm')
            recipe=plan['training_plans'][arm]
            if manifest['plan_sha256']!=sha(ROOT/recipe):raise ValueError('Training recipe mismatch')
            h=sha(folder/'best.npz')
            if h!=best['npz_sha256'] or h!=manifest['best_npz_sha256']:raise ValueError('Selected model mismatch')
            models[arm]={'network':h,'metadata':sha(folder/'best.json')}
        (out/'selected-models.json').write_text(json.dumps(models,indent=2)+'\n')
        def verify_models():
            verify()
            for arm,directory in plan['models'].items():
                if sha(ROOT/directory/'best.npz')!=models[arm]['network'] or sha(ROOT/directory/'best.json')!=models[arm]['metadata']:
                    raise ValueError('Selected model changed')
        state.update(stage='benchmarking_forward_cost');save();verify_models()
        run([sys.executable,'scripts/benchmark_residual_inference.py','--models',
             *plan['models'].values(),'--records',plan['benchmark_records'],
             '--out',str(out/'inference-cost.json')],out/'inference-cost.log')
        verify_models()
        for match in plan['matches']:
            verify_models();state.update(stage='playing',match=match['id']);save()
            candidate=plan['models'][match['candidate']]+'/best.npz'
            baseline=match['baseline'];neural=baseline!='teacher'
            cmd=[sys.executable,'scripts/native_match.py','--candidate=adapters/neural_policy_gpu_uci.sh',
                '--candidate-net='+candidate,'--candidate-options=LeafBatch=32',
                '--baseline='+('adapters/neural_policy_gpu_uci.sh' if neural else plan['teacher_binary']),
                '--net='+(plan['models'][baseline]+'/best.npz' if neural else plan['teacher_net']),
                '--tc=10+0.1','--games=64','--seed='+str(match['seed']),
                '--concurrency=2','--stage=development','--out='+str(out/match['id'])]
            if neural:cmd+=['--baseline-options=LeafBatch=32']
            run(cmd,out/(match['id']+'.log'));verify_models()
            run([sys.executable,'scripts/audit_native_games.py','--run',str(out/match['id']),
                 '--out',str(out/(match['id']+'-audit.json'))],out/(match['id']+'-audit.log'))
            summary=json.loads((out/match['id']/'cdca-summary.json').read_text())
            audit=json.loads((out/(match['id']+'-audit.json')).read_text())
            state['matches'].append({'id':match['id'],'result':summary,'audit':audit});save()
        verify_models();state.update(stage='completed');save()
    except BaseException as error:
        state.update(stage='failed',error=str(error));save();raise


if __name__=='__main__':main()
