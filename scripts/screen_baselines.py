"""Opponent-only paired screening with independent referee and explicit profiles."""
import argparse,collections,hashlib,json,math,os,random,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summary(games,voids,expected):
    pairs=collections.defaultdict(list);scores=[];technical=[]
    for game in games:
        score=game['result'] if game['red']=='candidate' else 1-game['result']
        scores.append(score);pairs[game['pair']].append((game['red'],score))
        if game['reason'].startswith(('TIME_FORFEIT','illegal','MAX_PLIES','engine failure')):
            technical.append(game['reason'])
    values=[statistics.mean(s for _,s in rows) for rows in pairs.values()
        if len(rows)==2 and {side for side,_ in rows}=={'candidate','baseline'}]
    interval=None;bounded=None
    if values:
        rng=random.Random(20261026)
        boot=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(10000))
        interval=[boot[249],boot[9749]]
        radius=math.sqrt(math.log(40)/(2*len(values)));mean=statistics.mean(values)
        bounded=[max(0,mean-radius),min(1,mean+radius)]
    return {'games':len(games),'expected_games':expected,'wins':scores.count(1),
        'draws':scores.count(.5),'losses':scores.count(0),
        'score':statistics.mean(scores) if scores else None,'complete_pairs':len(values),
        'paired_bootstrap_95':interval,'paired_hoeffding_95':bounded,
        'void_attempts':len(voids),'technical_results':technical,
        'valid_technical_run':len(games)==expected and len(values)==expected//2 and not voids and not technical,
        'promotion_supported':False,'single_match_superiority_gate':False,
        'warning':'Opponent-only shortlist screening, not our-model strength or a transitive victory.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
    digest=sha(a.plan);plan=json.loads(Path(a.plan).read_text());out=ROOT/plan['out']
    if out.exists():raise ValueError('Refuse to overwrite or restart screening')
    def verify():
        if sha(a.plan)!=digest:raise ValueError('Screening plan changed')
        for name,expected in plan['sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Frozen input changed: '+name)
        node=os.environ.get('UNIVERSE_NODE')
        if not node or sha(node)!=plan['node_runtime_sha256']:raise ValueError('Node runtime mismatch')
        referee=ROOT/'external/devinokeefe-jieqi'
        if subprocess.check_output(['git','-C',str(referee),'rev-parse','HEAD'],text=True).strip()!='331a6e612a038c4a04cf980f7b9482910fd50078' or subprocess.check_output(['git','-C',str(referee),'diff','HEAD'],text=True).strip():
            raise ValueError('Require unchanged pinned external referee')
    verify();out.mkdir(parents=True)
    state={'schema':1,'pid':os.getpid(),'status':'running','plan_sha256':digest,'completed':[]}
    def save():
        tmp=out/'state.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'state.json')
    env=dict(os.environ,JIEQI_PYTHON=sys.executable,
        YUAN_NATIVE_DIR=str(ROOT/'build/yuan-native/source'),
        JIEQI_RULES_BACKEND='cython',JIEQI_LEGAL_BACKEND='cython',
        UNIVERSE_JIEQI_DIR=str(ROOT/'external/universe-jieqi'))
    if 'UNIVERSE_NODE' not in env:raise ValueError('Set a compatible Node runtime before screening')
    def run(command,log):
        with log.open('x') as handle:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save();code=child.wait();state['child_pid']=None
            return code
    save()
    try:
        for match in plan['matches']:
            verify();state['stage']=match['id'];save();folder=out/match['id'];folder.mkdir()
            left=plan['engines'][match['left']];right=plan['engines'][match['right']]
            command=[sys.executable,'scripts/referee_entry.py','--warmup','--timing-log',str(folder/'timing.jsonl')]
            for label,engine in [('candidate',left),('baseline',right)]:
                if engine['strict']:command+=['--strict-name',label]
                options=list(engine.get('options',[]))
                if engine.get('net'):options.insert(0,'EvalFile='+str(ROOT/engine['net']))
                specification=label+'='+str(ROOT/engine['binary'])
                if options:specification+=':'+','.join(options)
                command+=['--engine',specification]
            command+=['--games',str(match['games']),'--tc',plan['clock'],'--seed',str(match['seed']),
                '--concurrency','2','--threads','1','--hash','64','--scores','--out',str(folder)]
            hashes={'candidate':sha(ROOT/left['binary']),'baseline':sha(ROOT/right['binary']),
                'candidate_net':sha(ROOT/left['net']) if left.get('net') else None,
                'net':sha(ROOT/right['net']) if right.get('net') else None}
            manifest={'status':'running','referee_commit':'331a6e612a038c4a04cf980f7b9482910fd50078',
                'rules':plan['rules'],'hashes':hashes,'backbone_disclosure':'Opponent-only screening: neither actor is our trained candidate. Left is recorded as candidate solely for referee/archiver compatibility.',
                'args':{'candidate':left['binary'],'baseline':right['binary'],'games':match['games'],
                    'seed':match['seed'],'concurrency':2,'tc':plan['clock'],'movetime':100,
                    'stage':'development','candidate_options':','.join(left.get('options',[]))}}
            (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            code=run(command,out/(match['id']+'.log'));verify()
            if code:
                manifest['status']='failed';(folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                state['completed'].append({'id':match['id'],'status':'unverified_process_failure'});save();continue
            games=[json.loads(line) for line in (folder/'games.jsonl').read_text().splitlines()]
            voids=[json.loads(line) for line in (folder/'void.jsonl').read_text().splitlines()]
            result=summary(games,voids,match['games']);manifest['status']='completed'
            (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            (folder/'cdca-summary.json').write_text(json.dumps(result,indent=2)+'\n')
            prefix='results/20261026-baseline-screen-'+match['id']
            if run([sys.executable,'scripts/archive_native_results.py','--out',prefix+'.json',
                    '--raw-dir','results/raw',str(folder)],out/(match['id']+'-archive.log')):
                raise RuntimeError('Archiving failed')
            audit=None
            if result['valid_technical_run']:
                if run([sys.executable,'scripts/audit_native_games.py','--run',str(folder),
                        '--out',prefix+'-audit.json'],out/(match['id']+'-audit.log')):
                    raise RuntimeError('Replay audit failed')
                audit=json.loads((ROOT/(prefix+'-audit.json')).read_text())
            eligible=bool(audit and not audit['technical_games'] and not audit['clock_ambiguous_moves_from_rounding'])
            state['completed'].append({'id':match['id'],'left':match['left'],'right':match['right'],
                'result':result,'audit':audit,'eligible_for_screening':eligible});save()
        verify();state['status']='completed';save()
    except BaseException as error:
        state.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
