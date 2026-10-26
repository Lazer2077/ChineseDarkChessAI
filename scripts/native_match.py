"""Frozen paired match over the independent jqe referee and full private histories."""
import argparse, collections, hashlib, json, math, os, platform, random, statistics, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--baseline',required=True);p.add_argument('--net',required=True);p.add_argument('--candidate-net');p.add_argument('--candidate-options',default='');p.add_argument('--baseline-strict',action='store_true');p.add_argument('--baseline-options',default='');p.add_argument('--baseline-classical',action='store_true');p.add_argument('--out',required=True);p.add_argument('--games',type=int,default=32);p.add_argument('--movetime',type=int,default=100);p.add_argument('--tc',default='');p.add_argument('--seed',type=int,required=True);p.add_argument('--concurrency',type=int,default=2);p.add_argument('--stage',choices=['development','confirmation'],default='development');a=p.parse_args()
if a.games<2 or a.games%2 or a.movetime<1:raise SystemExit('Require even games and positive clock')
if sys.version_info<(3,11):raise SystemExit('Native referee requires Python 3.11+; choose a compatible interpreter')
root=Path(__file__).resolve().parents[1];src=root/'external/devinokeefe-jieqi';out=Path(a.out).resolve()
if out.exists():raise SystemExit('Refusing to reuse match directory')
paths={k:Path(v).resolve() for k,v in {'candidate':a.candidate,'baseline':a.baseline,'net':a.net,'candidate_net':a.candidate_net or a.net}.items()}
for path in paths.values():
    if not path.is_file():raise SystemExit(f'Missing artifact: {path}')
sha=subprocess.check_output(['git','-C',src,'rev-parse','HEAD'],text=True).strip()
if sha!='331a6e612a038c4a04cf980f7b9482910fd50078':raise SystemExit('Unexpected referee revision')
if subprocess.check_output(['git','-C',src,'diff','HEAD'],text=True).strip():raise SystemExit('Referee source modified')
out.mkdir(parents=True)
A=f"candidate={paths['candidate']}:EvalFile={paths['candidate_net']}"+((','+a.candidate_options) if a.candidate_options else '')
B=f"baseline={paths['baseline']}"
opts=[] if a.baseline_classical else [f"EvalFile={paths['net']}"]
if a.baseline_options:opts.append(a.baseline_options)
if opts:B+=':'+','.join(opts)
cmd=[sys.executable,root/'scripts/referee_entry.py','--warmup','--timing-log',out/'timing.jsonl',*(['--strict-name','baseline'] if a.baseline_strict else []),'--engine',A,'--engine',B,'--games',str(a.games),*(['--tc',a.tc] if a.tc else ['--movetime',str(a.movetime)]),'--seed',str(a.seed),'--concurrency',str(a.concurrency),'--threads','1','--hash','64','--scores','--out',out]
manifest={'args':vars(a),'command':list(map(str,cmd)),'host':platform.node(),'referee_commit':sha,'rules':'jqe pinned native: 80-ply no-capture, repetition/check/chase; complete observer-specific history','hashes':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},'status':'running','stage':a.stage,'referee_entry_sha256':hashlib.sha256((root/'scripts/referee_entry.py').read_bytes()).hexdigest(),'backbone_disclosure':'Candidate may reuse jqe; teacher network belongs to upstream. Only identified changes are our contribution.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
try:
    subprocess.run(list(map(str,cmd)),check=True,cwd=root)
    games=[json.loads(line) for line in (out/'games.jsonl').read_text().splitlines()];voids=[json.loads(line) for line in (out/'void.jsonl').read_text().splitlines()]
    pairs=collections.defaultdict(list);wins=draws=losses=0;technical=[]
    for g in games:
        score=g['result'] if g['red']=='candidate' else 1-g['result'];pairs[g['pair']].append((g['red']=='candidate',score));wins+=score==1;draws+=score==.5;losses+=score==0
        if g['reason'].startswith(('illegal','TIME_FORFEIT','MAX_PLIES')):technical.append(g['reason'])
    complete=[statistics.mean(s for _,s in group) for group in pairs.values() if len(group)==2 and {red for red,_ in group}=={True,False}]
    ci=None
    if len(complete)>=20:
        rng=random.Random(20261026);boot=sorted(statistics.mean(rng.choices(complete,k=len(complete))) for _ in range(10000));ci=[boot[249],boot[9749]]
    bounded=None
    if complete:
        mean=statistics.mean(complete);radius=math.sqrt(math.log(40)/(2*len(complete)))
        bounded=[max(0.,mean-radius),min(1.,mean+radius)]
    valid=len(games)==a.games and len(complete)==a.games//2 and not technical and not voids
    timings=[json.loads(line) for line in (out/'timing.jsonl').read_text().splitlines()] if (out/'timing.jsonl').exists() else []
    audit={}
    for name in ['candidate','baseline']:
        elapsed=sorted(t['elapsed_ms'] for t in timings if t['engine']==name and not t['warmup'])
        if elapsed:audit[name]={'moves':len(elapsed),'mean_ms':statistics.mean(elapsed),'p95_ms':elapsed[int(.95*(len(elapsed)-1))],'max_ms':max(elapsed)}
    report={'timing':audit,'game_clock':a.tc or None,'games':len(games),'expected_games':a.games,'wins':wins,'draws':draws,'losses':losses,'score':(wins+.5*draws)/len(games) if games else None,'complete_pairs':len(complete),'paired_bootstrap_95':ci,'paired_hoeffding_95':bounded,'void_attempts':len(voids),'technical_results':technical,'valid_technical_run':valid,'single_match_superiority_gate':a.stage=='confirmation' and valid and len(complete)>=100 and ci is not None and ci[0]>.5 and bounded[0]>.5,'promotion_supported':False,'warning':'A single development match is not a promotion or an all-engine superiority claim.'}
    (out/'cdca-summary.json').write_text(json.dumps(report,indent=2)+'\n');manifest['status']='completed';print(json.dumps(report),flush=True)
except BaseException as error:
    manifest['status']='failed';manifest['error']=str(error);raise
finally:(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
