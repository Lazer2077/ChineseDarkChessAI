"""Fixed current-actor self-play iterations, with separate strength diagnostics.

Research actors advance by the fixed validation recipe, even if a diagnostic
match is negative. This is not an accepted/promoted benchmark model.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.neural_replay import save_json,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());digest=sha(a.plan);out=ROOT/plan['out']
    if out.exists():raise ValueError('Use fresh series output; never restart a live run')
    def verify():
        if sha(a.plan)!=digest:raise ValueError('Series recipe changed')
        for file,expected in plan['sha256'].items():
            if sha(ROOT/file)!=expected:raise ValueError('Frozen series source changed: '+file)
    verify();out.mkdir(parents=True)
    state={'schema':1,'status':'running','pid':os.getpid(),'plan_sha256':digest,'completed':0,'iterations':[]}
    save_json(out/'state.json',state)
    env=dict(os.environ,JIEQI_NEURAL_PYTHON=sys.executable)
    def run(stage,command,log):
        verify();state['stage']=stage
        with log.open('x') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save_json(out/'state.json',state)
            code=child.wait()
        state['child_pid']=None
        if code:raise RuntimeError(stage+' exited '+str(code))
        verify()
    actor=plan['initial_model']
    try:
        for i in range(1,plan['iterations']+1):
            state['iteration']=i;iteration=out/f'iter-{i:03d}';iteration.mkdir()
            replay=iteration/'episodes';trained=iteration/'trained'
            pins={**plan['sha256'],str(Path(actor)/'best.npz'):sha(ROOT/actor/'best.npz'),str(Path(actor)/'best.json'):sha(ROOT/actor/'best.json')}
            generation={**plan['generation'],'schema':1,'out':str(replay.relative_to(ROOT)),'model':actor,
                'seed':plan['generation_seed']+i*10000,'games':plan['training_games']+plan['validation_games'],'sha256':pins}
            gp=iteration/'generation-plan.json';save_json(gp,generation)
            training={**plan['training'],'schema':1,'out':str(trained.relative_to(ROOT)),
                'initial_model':actor,'generation_plan':str(gp.relative_to(ROOT)),
                'training_games':list(range(plan['training_games'])),
                'selection_validation_games':list(range(plan['training_games'],generation['games'])),
                'seed':plan['training_seed']+i,'sha256':{**pins,str(gp.relative_to(ROOT)):sha(gp)}}
            tp=iteration/'training-plan.json';save_json(tp,training)
            run('generation',[sys.executable,'scripts/generate_neural_selfplay.py','--plan',str(gp)],iteration/'generation.log')
            run('training',[sys.executable,'scripts/train_neural_selfplay.py','--plan',str(tp)],iteration/'training.log')
            m=json.loads((trained/'manifest.json').read_text());best=json.loads((trained/'best.json').read_text())
            if m['status']!='completed' or m['epochs_completed']!=training['epochs'] or m['plan_sha256']!=sha(tp) or best['npz_sha256']!=sha(trained/'best.npz'):
                raise ValueError('Training recipe or model incomplete')
            candidate=str(trained.relative_to(ROOT));old=actor
            item={'iteration':i,'previous_actor':old,'research_actor':candidate,'network_sha256':best['npz_sha256'],
                'training_manifest_sha256':sha(trained/'manifest.json'),'generation_plan_sha256':sha(gp),'training_plan_sha256':sha(tp),
                'actor_update':'Fixed validation-selected research actor advances, independent of match outcomes. No benchmark promotion.', 'matches':{}}
            # Both screens finish at every iteration. Results never choose the next actor.
            for opponent in ['previous','teacher']:
                baseline='adapters/neural_policy_gpu_uci.sh' if opponent=='previous' else plan['teacher_binary']
                baseline_net=str(Path(old)/'best.npz') if opponent=='previous' else plan['teacher_net']
                match=iteration/('vs-'+opponent)
                command=[sys.executable,'scripts/native_match.py','--candidate=adapters/neural_policy_gpu_uci.sh',
                    '--candidate-net='+str(Path(candidate)/'best.npz'),'--candidate-options=LeafBatch='+str(plan['match_leaf_batch']),
                    '--baseline='+baseline,'--net='+baseline_net,'--tc='+plan['clock'],
                    '--games='+str(plan['match_games']),'--seed='+str(plan['match_seed']+i*10000+(0 if opponent=='previous' else 1000)),
                    '--concurrency='+str(plan['match_concurrency']),'--stage=development','--out='+str(match)]
                if opponent=='previous':command+=['--baseline-options=LeafBatch='+str(plan['match_leaf_batch'])]
                run('match-'+opponent,command,iteration/('vs-'+opponent+'.log'))
                item['matches'][opponent]=json.loads((match/'cdca-summary.json').read_text())
                if sha(trained/'best.npz')!=best['npz_sha256']:raise ValueError('Played candidate changed')
            actor=candidate;state['iterations'].append(item);state['completed']=i;state['research_actor']=actor
            save_json(out/'state.json',state)
        verify();state.update(status='completed',stage='finished');save_json(out/'state.json',state)
    except BaseException as error:
        state.update(status='failed',error=str(error));save_json(out/'state.json',state);raise

if __name__=='__main__':main()
