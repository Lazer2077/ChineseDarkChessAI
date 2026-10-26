"""Replay a completed native match from its seeded deals, including rounded actual clocks."""
import argparse,hashlib,json,random,subprocess,sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];source=root/'external/devinokeefe-jieqi';run=Path(a.run)
if subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()!='331a6e612a038c4a04cf980f7b9482910fd50078' or subprocess.check_output(['git','-C',source,'diff','HEAD'],text=True).strip():raise SystemExit('Require unchanged pinned referee')
sys.path.insert(0,str(source))
from jieqi.native import GameStatus,NativeGame,NativeRules
from jieqi.setup import Setup,Color,Kind,square,Piece
library=source/'build/release/lib/libjieqi_rules.so';rules=NativeRules(library)
meta=json.loads((run/'manifest.json').read_text());summary=json.loads((run/'cdca-summary.json').read_text());raw=(run/'games.jsonl').read_bytes();games=[json.loads(x) for x in raw.splitlines()]
if meta['status']!='completed' or len(games)!=meta['args']['games']:raise SystemExit('Incomplete match')
rng=random.Random(meta['args']['seed']);seeds=[str(rng.getrandbits(63)) for _ in range(len(games)//2)]
seen=set();moves=technical=ambiguous=clock_checks=0;minimum_lower=None;scores=[]
tc=meta['args'].get('tc');clock=None
if tc:
 base,_,inc=tc.partition('+');clock=(round(float(base)*1000),round(float(inc or 0)*1000))
for g in games:
 pair=g['pair'];key=(pair,g['red'])
 if key in seen or g['seed']!=seeds[pair] or {g['red'],g['black']}!={'candidate','baseline'}:raise ValueError('Duplicate or incorrect paired layout')
 seen.add(key);truth=list(Setup.from_seed(int(g['seed'])).board)
 public=''.join('.' if x is None else x.code() for x in truth)
 lower={Color.RED:clock[0],Color.BLACK:clock[0]} if clock else None;upper=dict(lower) if clock else None
 with NativeGame(rules,public,Color.RED) as game:
  for rec in g['moves']:
   snap=game.snapshot()
   if snap.status is not GameStatus.ONGOING:raise ValueError('Move after referee terminal')
   o,d=square(rec['m'][:2]),square(rec['m'][2:4]);mover,victim=truth[o],truth[d]
   if mover is None or mover.color is not snap.turn:raise ValueError('Wrong mover')
   reveal=mover.kind if mover.hidden else None
   if rec.get('r')!=(reveal.value if reveal else None) or rec.get('c')!=(victim.kind.value if victim else None):raise ValueError('Recorded reveal/capture differs from seeded truth')
   game.advance(rec['m'],reveal);truth[d]=Piece(mover.color,mover.kind,False);truth[o]=None;moves+=1
   if clock and not rec.get('rand'):
    t=rec['t']
    if not isinstance(t,int) or t<0:raise ValueError('Missing rounded actual wall time')
    side=snap.turn;lower[side]-=t+.5;upper[side]-=max(0,t-.5)
    if upper[side]<0:raise ValueError('A completed legal move definitely exceeded its game clock')
    ambiguous+=lower[side]<0;clock_checks+=1;minimum_lower=lower[side] if minimum_lower is None else min(minimum_lower,lower[side])
    lower[side]+=clock[1];upper[side]+=clock[1]
  terminal=game.snapshot()
  is_technical=g['reason'].startswith(('TIME_FORFEIT','illegal','MAX_PLIES','engine failure'))
  if is_technical:technical+=1
  else:
   expected={GameStatus.DRAW:.5,GameStatus.RED_LOSES:0.,GameStatus.BLACK_LOSES:1.}.get(terminal.status)
   if expected is None or g['result']!=expected or g['reason']!=terminal.status.name:raise ValueError('Terminal outcome or reason differs from referee replay')
 if g['plies']!=len(g['moves']):raise ValueError('Ply count mismatch')
 scores.append(g['result'] if g['red']=='candidate' else 1-g['result'])
for i in range(len(seeds)):
 if (i,'candidate') not in seen or (i,'baseline') not in seen:raise ValueError('Missing color swap')
wdl=[scores.count(x) for x in [1,.5,0]]
if wdl!=[summary[k] for k in ['wins','draws','losses']]:raise ValueError('WDL does not match raw games')
report={'schema':1,'run':run.name,'games':len(games),'complete_pairs':len(seeds),'moves_replayed':moves,'wins_draws_losses':wdl,'seeded_identities_match':True,'all_recorded_moves_legal':True,'nontechnical_terminal_results_match':True,'technical_games':technical,'actual_clock_moves_checked':clock_checks,'clock_ambiguous_moves_from_rounding':int(ambiguous),'minimum_clock_lower_margin_ms':minimum_lower,'raw_games_sha256':hashlib.sha256(raw).hexdigest(),'replay_referee_library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'scope':'Post-match replay using pinned native referee; t fields are measured wall times rounded to milliseconds. Each clock check allows +/-0.5ms measurement rounding; ambiguous margins are reported, never silently certified. This verifies records, not statistical superiority or all-engine coverage.'}
out=Path(a.out)
if out.exists():raise SystemExit('Refuse to overwrite an audit')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
