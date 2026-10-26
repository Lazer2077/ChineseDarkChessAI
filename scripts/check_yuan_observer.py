"""Cross-check observer-only reconstruction against seeded independent native games."""
import argparse,collections,hashlib,json,random,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--games',type=int,default=8);p.add_argument('--plies',type=int,default=80)
p.add_argument('--seed',type=int,default=20261026);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'external/devinokeefe-jieqi'))
from jieqi.native import NativeGame,NativeRules,GameStatus
from jieqi.setup import Color,Kind,Setup,square,Piece
out=Path(a.out)
if out.exists():raise SystemExit('Use fresh report')
rules=NativeRules(root/'external/devinokeefe-jieqi/build/release/lib/libjieqi_rules.so')
proc=subprocess.Popen([sys.executable,str(root/'adapters/yuan_observer.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
def query(tokens,viewer,seed=0):
    proc.stdin.write(json.dumps({'tokens':tokens,'viewer':viewer,'placeholder_seed':seed})+'\n');proc.stdin.flush()
    row=proc.stdout.readline()
    if not row:raise RuntimeError('Observer process exited')
    return json.loads(row)
initial=collections.Counter({'R':2,'N':2,'B':2,'A':2,'C':2,'P':5})
rng=random.Random(a.seed);positions=private=unknown_events=invariance=privacy_refusals=0;differences=[]
try:
 for g in range(a.games):
    truth=list(Setup.from_seed(rng.getrandbits(63)).board)
    public=''.join('.' if pc is None else ('X' if pc.color==Color.RED else 'x') if pc.hidden else pc.code() for pc in truth)
    tokens={c:[] for c in Color};revealed={c:collections.Counter() for c in Color};captured={c:collections.Counter() for c in Color};lost={c:0 for c in Color}
    with NativeGame(rules,public,Color.RED) as game:
     for ply in range(a.plies):
      snap=game.snapshot()
      if snap.status is not GameStatus.ONGOING:break
      side=snap.turn;viewer=1 if side==Color.RED else 0
      reply=query(tokens[side],viewer)
      if not reply['ok']:raise ValueError(f'game {g} ply {ply}: '+reply['error'])
      view=reply['snapshot'];positions+=1
      if view['board']!=snap.board:raise ValueError('Public board mismatch')
      if view['side']!=('r' if side==Color.RED else 'b'):raise ValueError('Acting side mismatch')
      if view['no_capture_plies']!=snap.no_capture_plies:raise ValueError('No-capture counter mismatch')
      for color in [Color.BLACK,Color.RED]:
       idx=0 if color==Color.BLACK else 1
       expected=initial-revealed[color]
       if color!=side:expected-=captured[side]
       actual=[view['known'][idx][i] for i in range(2,8)]
       if actual!=[expected[k] for k in 'RNBACP']:raise ValueError('Observer pool mismatch')
       if view['unknown'][idx]!=(lost[color] if color==side else 0):raise ValueError('Unknown capture count mismatch')
       if [view['public'][idx][i] for i in range(2,8)]!=[(initial-revealed[color])[k] for k in 'RNBACP']:
        raise ValueError('Public pool leaked private identities')
      if ply%10==0:
       other=query(tokens[side],viewer,991)['snapshot']
       if view!=other:raise ValueError('Arbitrary hidden-past sampling affected observation or legal set')
       invariance+=1
      legal=sorted(rules.legal_moves(snap.board,side))
      if view['legal_moves']!=legal:
       differences.append({'game':g,'ply':ply,'yuan_only':sorted(set(view['legal_moves'])-set(legal)),'native_only':sorted(set(legal)-set(view['legal_moves']))})
      move=rng.choice(legal);frm,to=square(move[:2]),square(move[2:4]);mover,victim=truth[frm],truth[to]
      reveal=mover.kind if mover.hidden else None
      cap=victim.kind if victim and victim.hidden else None
      def letter(kind,color):return kind.value if color==Color.RED else kind.value.lower()
      base=move[:4];suffix=letter(reveal,side) if reveal else ''
      tokens[side].append(base+suffix+(letter(cap,victim.color) if cap else ''))
      tokens[side.opponent].append(base+suffix)
      if reveal:revealed[side][reveal.value]+=1
      if cap:
       captured[side][cap.value]+=1;lost[victim.color]+=1;private+=1;unknown_events+=1
       leaked=tokens[side.opponent].copy();leaked[-1]+=letter(cap,victim.color)
       if query(leaked,1 if side.opponent==Color.RED else 0)['ok']:
        raise ValueError('Opponent capture secret accepted')
       missing=tokens[side].copy();missing[-1]=missing[-1][:-1]
       if query(missing,1 if side==Color.RED else 0)['ok']:
        raise ValueError('Missing own capture identity accepted')
       privacy_refusals+=2
      game.advance(base,reveal);truth[to]=Piece(side,mover.kind,False);truth[frm]=None
 # Hand-created malformed/secret suffixes must never be accepted as knowledge.
 for request in [(['a3a4'],1),(['a3a4Pa6a5pr'],1),(['z3a4P'],1)]:
    if query(*request)['ok']:raise ValueError('Malformed history accepted')
finally:
 proc.stdin.close();proc.wait(timeout=5)
report={'schema':1,'games':a.games,'max_plies':a.plies,'seed':a.seed,'positions_checked':positions,
 'private_captures':private,'unknown_capture_events':unknown_events,'placeholder_invariance_checks':invariance,'private_suffix_faults_refused':privacy_refusals,
 'public_board_and_actor_inventory_match':True,'public_inventory_has_no_private_capture_deductions':True,
 'rules_backend':view['rules_backend'],'legal_backend':view['legal_backend'],
 'legal_set_differences':differences,'observer_sha256':hashlib.sha256((root/'adapters/yuan_observer.py').read_bytes()).hexdigest(),
 'scope':'Observer-only cross-check on sampled games. Legal differences remain explicit; no full rule compatibility, timed search, neural-model availability or strength claim.'}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
