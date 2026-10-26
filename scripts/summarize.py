"""Report paired scores without silently dropping incomplete games."""
import argparse, json, math, random, statistics
from collections import defaultdict
from pathlib import Path

def summarize(games, seed=1, samples=10000):
    pairs=defaultdict(list); w=d=l=truncated=0
    for g in games:
        r=g['result']
        if r.get('truncated'):
            truncated+=1;score=None
        elif r['winner'] is None:
            d+=1;score=.5
        elif r['winner']==g['candidate']:
            w+=1;score=1.
        else:
            l+=1;score=0.
        pairs[g['layoutSeed']].append((g['candidate'],score))
    complete=[statistics.mean(v for _,v in group) for group in pairs.values()
              if len(group)==2 and {c for c,_ in group}=={'r','b'} and all(v is not None for _,v in group)]
    formal=bool(games) and len(complete)==len(pairs) and len(games)==2*len(pairs)
    ci=None
    if len(complete)>=20:
        rng=random.Random(seed);boot=sorted(statistics.mean(rng.choices(complete,k=len(complete))) for _ in range(samples))
        ci=[boot[int(.025*(samples-1))],boot[int(.975*(samples-1))]]
    bounded=None
    if complete:
        mean=statistics.mean(complete);radius=math.sqrt(math.log(40)/(2*len(complete)))
        bounded=[max(0.,mean-radius),min(1.,mean+radius)]
    n=w+d+l
    return {'games':len(games),'wins':w,'draws':d,'losses':l,'truncated':truncated,
            'completed_score':(w+.5*d)/n if n else None,
            'score_bounds_including_unfinished':[(w+.5*d)/len(games),(w+.5*d+truncated)/len(games)] if games else None,
            'complete_pairs':len(complete),'all_pairs_complete':formal,'paired_bootstrap_95':ci,'paired_hoeffding_95':bounded,
            'superiority_gate':formal and len(complete)>=100 and ci is not None and ci[0]>.5 and bounded[0]>.5,
            'note':'Bootstrap uses complete layout pairs; missing pairs invalidate superiority. Equal-budget/rule compatibility and independent confirmation must be checked separately.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args()
    root=Path(a.run);games=json.loads((root/'summary.json').read_text())['results'];report=summarize(games)
    print(json.dumps(report,indent=2));(root/'statistics.json').write_text(json.dumps(report,indent=2)+'\n')
