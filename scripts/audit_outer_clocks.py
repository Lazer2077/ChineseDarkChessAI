"""Supplement rounded replay with a stricter high-resolution outer-clock witness.

For each fresh engine PID, the frozen wrapper measures an interval containing
the charged Engine.go interval. Sum these upper durations, plus a 1us guard per
call, from the exact initial bank and increments. A nonnegative lower bank after
every call certifies wall-clock safety without mapping PIDs to individual games.
All raw move counts by (engine,ply), warmup/PID lifetimes and clock fields must
also agree. This does not change existing rounded audits or their flags.
"""
import argparse,collections,gzip,hashlib,json,math,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
WRAPPER_SHA='c66d886f08e709b720f8907c6b23ab81c70a335bd69eb7f969a7c62658dd2828'


def certify(games,timings,base_ms,increment_ms,guard_ms=.001):
    if not math.isfinite(guard_ms) or guard_ms<.001:raise ValueError('Require at least1us rounding guard')
    expected=collections.Counter()
    for game in games:
        for ply,move in enumerate(game['moves']):
            if move.get('rand'):raise ValueError('Random opening clock accounting not supported')
            expected[(game['red'] if ply%2==0 else game['black'],ply)]+=1
    actual=collections.Counter();warmups=collections.Counter();groups=collections.defaultdict(list)
    labels={}
    for row in timings:
        pid=row['pid'];label=row['engine']
        if type(pid)!=int or pid<=0 or label not in ('candidate','baseline'):
            raise ValueError('Invalid PID/engine')
        if pid in labels and labels[pid]!=label:raise ValueError('PID label changed')
        labels[pid]=label
        if row['warmup']:
            if groups[pid] or row['go']!='movetime 1':raise ValueError('Unexpected warmup/order')
            warmups[pid]+=1;continue
        if warmups[pid]!=1:raise ValueError('Missing or repeated warmup/PID reuse')
        elapsed=row['elapsed_ms']
        if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<0:
            raise ValueError('Invalid high-resolution duration')
        parts=row['go'].split()
        if len(parts)!=8 or parts[::2]!=['wtime','btime','winc','binc']:
            raise ValueError('Require real game-clock commands')
        numbers=list(map(int,parts[1::2]))
        if min(numbers)<0 or numbers[2:]!=[increment_ms,increment_ms]:
            raise ValueError('Clock profile changed')
        ply=row['ply']
        if type(ply)!=int or ply<0:raise ValueError('Invalid ply')
        groups[pid].append(row);actual[(label,ply)]+=1
    if len(warmups)!=2*len(games) or any(n!=1 for n in warmups.values()) or actual!=expected:
        raise ValueError('Missing/duplicate move calls or incorrect engine lifetimes')
    minimum=math.inf;ambiguous=[];calls=0
    for pid,rows in groups.items():
        bank=float(base_ms)
        if rows and [r['ply'] for r in rows]!=list(range(rows[0]['ply'],rows[0]['ply']+2*len(rows),2)):
            raise ValueError('Nonconsecutive per-engine history')
        if rows and rows[0]['ply'] not in (0,1):raise ValueError('Missing initial move call')
        for row in rows:
            bank-=row['elapsed_ms']+guard_ms;minimum=min(minimum,bank);calls+=1
            if bank<0:ambiguous.append({'engine':labels[pid],'pid':pid,'ply':row['ply'],'lower_bank_ms':bank})
            bank+=increment_ms
    return {'calls_checked':calls,'engine_processes':len(warmups),
        'move_call_counts_match':True,'single_game_pid_lifetimes_match':True,
        'guard_ms_per_call':guard_ms,'minimum_outer_bank_lower_ms':minimum,
        'ambiguous_calls':ambiguous,'certified_outer_wall_clock_safe':not ambiguous}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--out',required=True);p.add_argument('--raw-dir',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise ValueError('Refuse to overwrite an audit')
    plan=json.loads(Path(a.plan).read_text());run=Path(a.run)
    frozen=plan.get('sha256',plan.get('source_and_artifact_hashes',{}))
    if frozen.get('scripts/referee_entry.py')!=WRAPPER_SHA or sha(ROOT/'scripts/referee_entry.py')!=WRAPPER_SHA:
        raise ValueError('Require the known containing-interval wrapper')
    for name,h in frozen.items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen source changed: '+name)
    referee=ROOT/'external/devinokeefe-jieqi'
    if subprocess.check_output(['git','-C',str(referee),'rev-parse','HEAD'],text=True).strip()!='331a6e612a038c4a04cf980f7b9482910fd50078' or subprocess.check_output(['git','-C',str(referee),'diff','HEAD'],text=True).strip():
        raise ValueError('Require unchanged pinned referee source')
    manifest=json.loads((run/'manifest.json').read_text());summary=json.loads((run/'cdca-summary.json').read_text())
    if manifest['status']!='completed' or not summary['valid_technical_run']:
        raise ValueError('Never rescue a failed/void/incomplete/technical run')
    games=[json.loads(s) for s in (run/'games.jsonl').read_text().splitlines()]
    if len(games)!=manifest['args']['games']:raise ValueError('Incomplete games')
    raw=(run/'timing.jsonl').read_bytes();timings=[json.loads(s) for s in raw.splitlines()]
    base,_,increment=manifest['args']['tc'].partition('+')
    result=certify(games,timings,round(float(base)*1000),round(float(increment or 0)*1000))
    raw_dir=Path(a.raw_dir);raw_dir.mkdir(parents=True,exist_ok=True)
    target=raw_dir/('outer-clocks-'+run.name+'-'+hashlib.sha256(raw).hexdigest()[:12]+'.jsonl.gz')
    if target.exists():raise ValueError('Refuse to overwrite timing evidence')
    target.write_bytes(gzip.compress(raw,mtime=0))
    report={'schema':1,'run':run.name,'scope':'Supplemental conservative outer-wall witness; original rounded audit and preregistered eligibility remain unchanged. No statistical superiority or retroactive original-gate pass.',
        'plan':a.plan,'plan_sha256':sha(a.plan),'games_sha256':sha(run/'games.jsonl'),
        'timing_sha256':hashlib.sha256(raw).hexdigest(),'raw_timing':str(target),
        'wrapper_sha256':WRAPPER_SHA,'script_sha256':sha(__file__),**result}
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
