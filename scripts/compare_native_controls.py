"""Exploratory common-layout contrast of two candidates against one fixed opponent."""
import argparse,collections,gzip,hashlib,json,math,random,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--learner',action='append',required=True)
p.add_argument('--control',action='append',required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out)
if out.exists() or len(a.learner)!=len(a.control):raise SystemExit('Require fresh output and matched arm counts')
def load(filename):
    doc=json.loads(Path(filename).read_text())['reports'][0]
    if not doc['result']['valid_technical_run'] or doc['result']['void_attempts'] or doc['result']['technical_results']:raise ValueError('Require clean completed runs')
    raw=gzip.decompress(Path(doc['raw_games']['path']).read_bytes())
    if hashlib.sha256(raw).hexdigest()!=doc['raw_games']['uncompressed_sha256']:raise ValueError('Raw digest differs')
    groups=collections.defaultdict(list)
    for line in raw.splitlines():
        row=json.loads(line);groups[row['pair']].append(row)
    pairs={}
    for group in groups.values():
        if len(group)!=2 or {r['red'] for r in group}!={'candidate','baseline'} or len({r['seed'] for r in group})!=1:raise ValueError('Invalid color pair')
        pairs[group[0]['seed']]=statistics.mean(r['result'] if r['red']=='candidate' else 1-r['result'] for r in group)
    if len(pairs)*2!=doc['budget']['games']:raise ValueError('Missing or duplicated layout')
    return doc,pairs
def describe(strata):
    values=[x for s in strata for x in s];rng=random.Random(20261026)
    boot=sorted(statistics.mean(x for s in strata for x in rng.choices(s,k=len(s))) for _ in range(10000))
    mean=statistics.mean(values);radius=2*math.sqrt(math.log(40)/(2*len(values)))
    return {'layout_pairs':len(values),'score_difference':mean,'stratified_layout_bootstrap_95':[boot[249],boot[9749]],'range_two_hoeffding_95':[max(-1,mean-radius),min(1,mean+radius)]}
strata=[];sets=[];used=set()
for learnerfile,controlfile in zip(a.learner,a.control):
    learner,lp=load(learnerfile);control,cp=load(controlfile)
    if learner['budget']!=control['budget'] or learner['rules']!=control['rules'] or learner['source_referee_commit']!=control['source_referee_commit']:raise ValueError('Unmatched match conditions')
    for key in ['baseline','net','candidate']:
        if learner['hashes'][key]!=control['hashes'][key]:raise ValueError('Search binary or opponent differs')
    if learner['candidate_options']!=control['candidate_options'] or set(lp)!=set(cp) or used&set(lp):raise ValueError('Unmatched or reused layouts/options')
    used|=set(lp);deltas=[lp[seed]-cp[seed] for seed in sorted(lp)];strata.append(deltas)
    sets.append({'learner_report':learnerfile,'control_report':controlfile,'learner_raw_sha256':learner['raw_games']['uncompressed_sha256'],'control_raw_sha256':control['raw_games']['uncompressed_sha256'],**describe([deltas])})
report={'schema':1,'scope':'Exploratory opponent-specific common-layout contrast, not direct head-to-head strength or a general learned-improvement/causal claim. Sequential arm time order and shared-host scheduling are not randomized. Checkpoint and other model selection are not corrected by these intervals.',
 'method':'Subtract control score from learner score after averaging each color-swapped layout pair; bootstrap whole matched layouts independently within each fixed set, preserving stratum size. Difference range [-1,1], so conservative Hoeffding radius has range width 2.',
 'bootstrap_seed':20261026,'bootstrap_samples':10000,'sets':sets,'combined':describe(strata),'promotion_supported':False,
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
