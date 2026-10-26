"""UCI clock adapter for pinned unmodified Yuan root-consistent ISMCTS.

Only an iteration-boundary deadline is inserted into the imported pure search;
the MIT upstream checkout remains unchanged. No unpublished neural net is used.
"""
import hashlib,inspect,json,random,sys,time
from yuan_observer import load_sources,reconstruct

START_FEN='xxxxkxxxx/9/1x5x1/x1x1x1x1x/9/9/X1X1X1X1X/1X5X1/9/XXXXKXXXX w R2A2C2P5N2B2r2a2c2p5n2b2 0 1'

def timed_search_function():
    jq,mc=load_sources()
    original=mc.ris_ismcts
    text=inspect.getsource(original)
    needle='    for _ in range(iterations):\n'
    if text.count(needle)!=1:raise RuntimeError('Pinned iteration loop changed')
    patched=text.replace(needle,needle+'''        if _clock_deadline is not None and _clock_now() >= _clock_deadline and root_node.children:
            break
''')
    namespace=dict(mc.__dict__,_clock_deadline=None,_clock_now=time.perf_counter)
    exec(compile(patched,'<yuan-iteration-deadline>','exec'),namespace)
    return original,namespace['ris_ismcts'],namespace,hashlib.sha256(patched.encode()).hexdigest()

def main():
    jq,_=load_sources();_,search,namespace,patch_hash=timed_search_function()
    tokens=[];seed=20261026;overhead=20;max_iterations=1000000
    for line in sys.stdin:
        command=line.strip();parts=command.split()
        if command=='uci':
            print('id name Yuan root-ISMCTS (observer/clock adapter)')
            print('option name Seed type spin default 20261026 min 1 max 2147483647')
            print('option name MoveOverhead type spin default 20 min 0 max 1000')
            print('option name Iterations type spin default 1000000 min 1 max 1000000')
            print('info string iteration deadline sha256 '+patch_hash)
            print('uciok',flush=True)
        elif command=='isready':print('readyok',flush=True)
        elif command=='ucinewgame':tokens=[]
        elif command.startswith('setoption name '):
            name,_,value=command[len('setoption name '):].partition(' value ')
            if name=='Seed':seed=int(value)
            elif name=='MoveOverhead':overhead=int(value)
            elif name=='Iterations':max_iterations=max(1,int(value))
            # Single-threaded Python search: Threads/Hash have no supported knobs.
        elif command.startswith('position '):
            if parts[1]=='fen':
                end=parts.index('moves') if 'moves' in parts else len(parts)
                if ' '.join(parts[2:end])!=START_FEN:raise ValueError('Only the standard public start FEN is supported')
            elif parts[1]!='startpos':raise ValueError('Unsupported position')
            tokens=parts[parts.index('moves')+1:] if 'moves' in parts else []
        elif parts and parts[0]=='go':
            started=time.perf_counter()
            def number(key,default=None):return float(parts[parts.index(key)+1]) if key in parts else default
            bank=number('wtime' if len(tokens)%2==0 else 'btime')
            inc=number('winc' if len(tokens)%2==0 else 'binc',0)
            movetime=number('movetime')
            if movetime is None:movetime=min(bank/25+.75*inc,bank/4) if bank is not None else 1000
            allowance=max(1,movetime-overhead)
            namespace['_clock_deadline']=started+allowance/1000
            state,belief,_=reconstruct(tokens)
            rng=random.Random(seed+len(tokens)*1000003)
            move,tree=search(state,belief,iterations=max_iterations,rng=rng)
            print('info string '+json.dumps({'iterations':tree.visits,'elapsed_ms':(time.perf_counter()-started)*1000,'budget_ms':allowance}),flush=True)
            print('bestmove '+(jq.format_move_uci(move) if move else '0000'),flush=True)
        elif command=='quit':break

if __name__=='__main__':main()
