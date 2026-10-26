"""Resumable value-based expert iteration on the attributed native Jieqi backbone.

Actor = best network + chance-aware alpha-beta search; learner = outcome/search
value regression over recent actor games. No PPO, policy head or AlphaZero claim.
"""
import argparse,fcntl,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def retained_records(path):
    size=Path(path).stat().st_size
    if not size or size%112:raise ValueError('Empty or partial TrainRecord file')
    return size//112

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--iterations',type=int,default=2);p.add_argument('--games',type=int,default=8192);p.add_argument('--val-games',type=int,default=512);p.add_argument('--nodes',type=int,default=5000);p.add_argument('--workers',type=int,default=4);p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch',type=int,default=4096);p.add_argument('--gate-games',type=int,default=200);p.add_argument('--gate-tc',default='5+0.05');p.add_argument('--referee-python',required=True);p.add_argument('--window',type=int,default=2);a=p.parse_args()
    if min(a.iterations,a.games,a.val_games,a.nodes,a.workers,a.epochs,a.batch,a.window)<1 or a.gate_games<2 or a.gate_games%2:raise SystemExit('Invalid experiment size')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another live controller holds this run')
    actor=ROOT/'build/data-actor/bin/jqe';src=ROOT/'external/devinokeefe-jieqi';teacher=src/'engine/nets/it014.jqn';baseline=src/'engine/build/jqe';trainer=src/'engine/train/train.py'
    pins={'actor':sha(actor),'teacher':sha(teacher),'baseline':sha(baseline),'trainer':sha(trainer),'controller':sha(__file__)}
    state_path=out/'state.json';config=vars(a).copy();config['out']=str(out)
    state=json.loads(state_path.read_text()) if state_path.exists() else {'config':config,'pins':pins,'completed':0,'history':[],'replay':[],'best_net':str(teacher),'best_checkpoint':None,'status':'initialized'}
    if state['config']!=config or state['pins']!=pins:raise SystemExit('Run configuration/source differs; use a new experiment directory')
    # A missing controller alone does not prove its subprocess stopped.
    old_pid=state.get('child_pid')
    if old_pid and Path(f'/proc/{old_pid}/cmdline').exists():
        actual=Path(f'/proc/{old_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if str(out) in actual:raise SystemExit(f'Previous child {old_pid} is still live; do not restart it')
    state['controller_pid']=os.getpid();state['child_pid']=None
    def save():atomic(state_path,state)
    def command(stage,cmd,log):
        state['stage']=stage;state['status']='running';state['command']=list(map(str,cmd));save()
        print(stage,flush=True)
        with open(log,'w') as f:
            child=subprocess.Popen(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
            state['child_pid']=child.pid;save();code=child.wait()
        state['child_pid']=None;save()
        if code:raise RuntimeError(f'{stage} failed ({code}); inspect {log}')
    def generated(tag,games,seed,net):
        data=out/f'{tag}.bin';marker=out/f'{tag}.complete.json';log=out/f'{tag}.log'
        recipe={'actor':pins['actor'],'net':sha(net),'games':games,'seed':seed,'nodes':a.nodes,'workers':a.workers}
        if marker.exists():
            old=json.loads(marker.read_text())
            if old['recipe']!=recipe or sha(data)!=old['sha256']:raise ValueError('Completed data differs from recipe')
            retained_records(data);return data
        partial=out/f'{tag}.bin.part'
        if partial.exists():partial.rename(out/f'{tag}.aborted-{time.time_ns()}.bin')
        command(tag,[actor,'selfplay','games',games,'nodes',a.nodes,'threads',a.workers,'seed',seed,'random',8,'evalfile',net,'out',partial],log)
        text=log.read_text()
        done=re.search(r'selfplay done (\d+) games, (\d+) records',text);rejected=re.search(r'selfplay rejected (\d+)',text)
        if not done or int(done[1])!=games or not rejected:raise ValueError('Actor did not attest completed generation')
        n=retained_records(partial)
        if int(done[2])!=n:raise ValueError('Actor record count differs from file')
        if 'apply failed:' in text:raise ValueError('Observed-move transition failed; generation quarantined')
        partial.replace(data);atomic(marker,{'recipe':recipe,'sha256':sha(data),'records':n,'rejected_games':int(rejected[1])});return data
    try:
        if state['best_checkpoint'] is None:
            imported=out/'initial-import'
            if not (imported/'manifest.json').exists():
                if imported.exists():imported.rename(out/f'initial-import-aborted-{time.time_ns()}')
                command('initial-import',[sys.executable,ROOT/'scripts/import_jqn.py','--net',teacher,'--out',imported],out/'initial-import.log')
            if json.loads((imported/'manifest.json').read_text())['source_sha256']!=pins['teacher']:raise ValueError('Imported teacher mismatch')
            state['best_checkpoint']=str(imported/'warm-start.pt');save()
        val=generated('validation',a.val_games,9_000_000+a.seed*100,teacher)
        for iteration in range(state['completed']+1,a.iterations+1):
            prefix=f'iter-{iteration:03d}';state['iteration']=iteration;save()
            data=generated(prefix,a.games,8_000_000+a.seed*100+iteration,Path(state['best_net']))
            files=[*state['replay'],str(data)][-a.window:];model=out/f'{prefix}-model';marker=out/f'{prefix}-trained.json'
            recipe={'init':sha(state['best_checkpoint']),'datasets':{f:sha(f) for f in files},'validation':sha(val),'epochs':a.epochs,'seed':a.seed+iteration,'batch':a.batch}
            if marker.exists():
                old=json.loads(marker.read_text())
                if old['recipe']!=recipe or old['net_sha256']!=sha(model/'best.jqn'):raise ValueError('Trained candidate changed')
            else:
                if model.exists():model.rename(out/f'{prefix}-model-aborted-{time.time_ns()}')
                command(prefix+'-train',[sys.executable,trainer,*files,'--val-file',val,'--out',model,'--init',state['best_checkpoint'],'--epochs',a.epochs,'--batch',a.batch,'--lr',.00002,'--lam',.7,'--k',430,'--seed',a.seed+iteration,'--data-on-gpu'],out/f'{prefix}-train.log')
                atomic(marker,{'recipe':recipe,'net_sha256':sha(model/'best.jqn'),'checkpoint_sha256':sha(model/'best.pt')})
            gate=out/f'{prefix}-gate'
            if not (gate/'cdca-summary.json').exists():
                if gate.exists():gate.rename(out/f'{prefix}-gate-aborted-{time.time_ns()}')
                command(prefix+'-gate',[a.referee_python,ROOT/'scripts/native_match.py','--candidate',baseline,'--baseline',baseline,'--candidate-net',model/'best.jqn','--net',state['best_net'],'--tc',a.gate_tc,'--games',a.gate_games,'--seed',10_000_000+a.seed*100+iteration,'--out',gate],out/f'{prefix}-gate.log')
            report=json.loads((gate/'cdca-summary.json').read_text());boot=report.get('paired_bootstrap_95');bound=report.get('paired_hoeffding_95')
            accepted=report['valid_technical_run'] and report['complete_pairs']>=100 and bool(boot) and boot[0]>.5 and bool(bound) and bound[0]>.5
            entry={'iteration':iteration,'generated':str(data),'replay_files':files,'candidate':str(model/'best.jqn'),'internal_actor_accepted':accepted,'gate':report,'records_this_iteration':retained_records(data),'previous_actor':state['best_net'],'note':'Internal selection only, not all-engine superiority or final promotion.'}
            state['history'].append(entry);state['replay'].append(str(data))
            if accepted:state['best_net']=str(model/'best.jqn');state['best_checkpoint']=str(model/'best.pt')
            state['completed']=iteration;save();print(json.dumps(entry),flush=True)
        state['status']='completed';state['stage']='finished';save()
    except BaseException as error:
        state['status']='failed';state['error']=str(error);save();raise
if __name__=='__main__':main()
