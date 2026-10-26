"""Replay failed original-engine observations, with stderr and exit codes retained.

This is fault diagnosis at fixed move budgets, never replacement match evidence.
"""
import argparse,hashlib,json,queue,subprocess,sys,threading,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--void',required=True);p.add_argument('--binary',required=True)
p.add_argument('--net',required=True);p.add_argument('--out',required=True);p.add_argument('--movetime',type=int,default=200)
p.add_argument('--timing-log',help='Recover each failed go command from its uniquely matched measured move prefix')
p.add_argument('--attempts',type=int,default=3);a=p.parse_args()
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'external/devinokeefe-jieqi'))
from jieqi.setup import Color,Setup,Piece,square
from jieqi.native import NativeRules,NativeGame,GameStatus
out=Path(a.out)
if out.exists():raise SystemExit('Use fresh diagnostic directory')
out.mkdir(parents=True);binary=Path(a.binary).resolve();net=Path(a.net).resolve();input_bytes=Path(a.void).read_bytes()
rules=NativeRules(root/'external/devinokeefe-jieqi/build/release/lib/libjieqi_rules.so')
fen='xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1'
report={'schema':1,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'net_sha256':hashlib.sha256(net.read_bytes()).hexdigest(),
 'input_void_sha256':hashlib.sha256(input_bytes).hexdigest(),'scope':'Independent diagnostic replay of observer-only failure prefixes at fixed move budgets, with original search/evaluation unchanged. Does not alter or replace frozen match attempts.','cases':[]}
timing_groups={}
if a.timing_log:
    timing_bytes=Path(a.timing_log).read_bytes();report['timing_input_sha256']=hashlib.sha256(timing_bytes).hexdigest()
    report['scope']='Independent diagnostic replay using the original failed go command, matched uniquely by prior measured moves. Search history/transposition tables and operating-system scheduling are not reproduced. Frozen match outcomes remain unchanged.'
    for line in timing_bytes.splitlines():
        record=json.loads(line)
        if record['engine']=='baseline' and not record['warmup']:timing_groups.setdefault(record['pid'],[]).append(record)
for case,row in enumerate(json.loads(line) for line in input_bytes.splitlines()):
    truth=list(Setup.from_seed(int(row['seed'])).board)
    public=''.join('.' if pc is None else ('X' if pc.color==Color.RED else 'x') if pc.hidden else pc.code() for pc in truth)
    viewer=Color.RED if row['red']=='baseline' else Color.BLACK;tokens=[]
    with NativeGame(rules,public,Color.RED) as game:
        for move in row['moves']:
            snap=game.snapshot();frm,to=square(move['m'][:2]),square(move['m'][2:4]);mover,victim=truth[frm],truth[to]
            if move['r']!=(mover.kind.value if mover.hidden else None) or move['c']!=(victim.kind.value if victim else None):raise ValueError('Failure prefix identities mismatch seeded deal')
            def letter(kind,color):return kind.value if color==Color.RED else kind.value.lower()
            token=move['m']+(letter(mover.kind,mover.color) if mover.hidden else '')
            if victim and victim.hidden and snap.turn==viewer:token+=letter(victim.kind,victim.color)
            tokens.append(token);game.advance(move['m'],mover.kind if mover.hidden else None)
            truth[to]=Piece(mover.color,mover.kind,False);truth[frm]=None
        if game.snapshot().turn!=viewer or game.snapshot().status is not GameStatus.ONGOING:raise ValueError('Failed engine not on move at a valid ongoing prefix')
    go_command='movetime '+str(a.movetime)
    if a.timing_log:
        expected=[(ply,move['t']) for ply,move in enumerate(row['moves']) if (ply%2==0)==(viewer==Color.RED)]
        matches=[]
        for records in timing_groups.values():
            if len(records)==len(expected)+1 and records[-1]['ply']==len(tokens) and all(r['ply']==ply and abs(r['elapsed_ms']-elapsed)<2 for r,(ply,elapsed) in zip(records,expected)):
                matches.append(records[-1]['go'])
        if len(matches)!=1:raise ValueError('Cannot uniquely recover failed clock command')
        go_command=matches[0]
    item={'pair':row['pair'],'seed':row['seed'],'viewer':viewer.value,'verified_prefix_plies':len(tokens),'observation_tokens':tokens,'go_command':go_command,'attempts':[]}
    for attempt in range(a.attempts):
        log=out/f'{case}-{attempt}.stderr';stdout=[];failure=None
        with log.open('w') as error:
            proc=subprocess.Popen([str(binary)],cwd=binary.parent,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error,text=True,bufsize=1)
            lines=queue.Queue()
            def read():
                for line in proc.stdout:lines.put(line.rstrip('\n'))
                lines.put(None)
            reader=threading.Thread(target=read,daemon=True);reader.start()
            def send(text):proc.stdin.write(text+'\n');proc.stdin.flush()
            def until(prefix,timeout=30):
                deadline=time.monotonic()+timeout
                while True:
                    left=deadline-time.monotonic()
                    if left<=0:raise queue.Empty()
                    line=lines.get(timeout=left)
                    if line is None:raise RuntimeError('process exited')
                    stdout.append(line)
                    if line.startswith(prefix):return line
            try:
                send('uci');until('uciok');send('setoption name Threads value 1');send('setoption name Hash value 64')
                send('setoption name EvalFile value '+str(net));send('isready');until('readyok')
                send('position startpos');send('go movetime 1');until('bestmove ')
                send('ucinewgame');send('isready');until('readyok')
                send('position fen '+fen+' moves '+' '.join(tokens));send('go '+go_command);answer=until('bestmove ',10)
            except (RuntimeError,queue.Empty,BrokenPipeError) as exc:failure=str(exc);answer=None
            finally:
                if proc.poll() is None:
                    try:send('quit');proc.wait(timeout=2)
                    except (BrokenPipeError,subprocess.TimeoutExpired):proc.kill();proc.wait()
                reader.join(timeout=1)
        (out/f'{case}-{attempt}.stdout').write_text('\n'.join(stdout)+'\n')
        stderr=log.read_text().replace(str(root),'<repo>')
        item['attempts'].append({'go_command':go_command,'returncode':proc.returncode,'failure':failure,'bestmove':answer,'stderr':stderr})
    report['cases'].append(item)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
