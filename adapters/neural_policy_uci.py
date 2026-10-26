"""CPU-first UCI research actor for the supervised CDCA policy/value models."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'adapters'))
import torch
from yuan_observer import reconstruct,load_sources
from yuan_uci import START_FEN
from research.neural_search import ObservationPUCT,load_model,TIMED_SEARCH_SHA256

def main():
    p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--device',default='cpu',choices=['cpu','cuda']);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    jq,_=load_sources();model_file=a.model;net=engine=None;metadata=None
    seed=20261029;overhead=20;leaf_batch=8;tokens=[]
    def ready():
        nonlocal net,engine,metadata
        if net is None:
            if not model_file:raise ValueError('Set EvalFile to a best.npz artifact with its best.json')
            path=Path(model_file);net,metadata=load_model(path.parent if path.is_file() else path,a.device)
            engine=ObservationPUCT(net,a.device,seed,leaf_batch)
    for line in sys.stdin:
        text=line.strip();parts=text.split()
        if text=='uci':
            print('id name CDCA residual policy/value information-set PUCT (research)')
            print('option name EvalFile type string default <empty>')
            print('option name Seed type spin default 20261029 min 1 max 2147483647')
            print('option name MoveOverhead type spin default 20 min 0 max 1000')
            print('option name LeafBatch type spin default 8 min 1 max 128')
            print('info string batch deadline sha256 '+TIMED_SEARCH_SHA256)
            print('uciok',flush=True)
        elif text.startswith('setoption name '):
            name,_,value=text[len('setoption name '):].partition(' value ')
            if name=='EvalFile':model_file=value;net=engine=None
            elif name=='Seed':seed=int(value)
            elif name=='MoveOverhead':overhead=int(value)
            elif name=='LeafBatch':leaf_batch=max(1,int(value));engine=None;net=None
            elif name=='Threads' and int(value)!=1:raise ValueError('This profile has exactly one CPU thread')
            # Hash is not a supported tree-memory knob; no equal hash claim.
        elif text=='isready':ready();print('readyok',flush=True)
        elif text=='ucinewgame':tokens=[]
        elif text.startswith('position '):
            if parts[1]=='fen':
                end=parts.index('moves') if 'moves' in parts else len(parts)
                if ' '.join(parts[2:end])!=START_FEN:raise ValueError('Only standard public start FEN supported')
            elif parts[1]!='startpos':raise ValueError('Unsupported position')
            incoming=parts[parts.index('moves')+1:] if 'moves' in parts else []
            # The pinned harness pads lone reveal suffixes for classical parsers.
            # This marker is not a hidden identity; the observer ledger still
            # determines whether a private capture suffix is required.
            tokens=[token[:-1] if token.endswith(('x','X')) else token for token in incoming]
        elif parts and parts[0]=='go':
            started=time.perf_counter();ready()
            def number(key,default=None):return float(parts[parts.index(key)+1]) if key in parts else default
            simulations=number('nodes');budget=number('movetime')
            if budget is None:
                bank=number('wtime' if len(tokens)%2==0 else 'btime')
                inc=number('winc' if len(tokens)%2==0 else 'binc',0)
                budget=min(bank/25+.75*inc,bank/4) if bank is not None else 1000
            engine.deadline=None if simulations else started+max(1,budget-overhead)/1000
            engine.rng.seed(seed+len(tokens)*1000003)
            state,belief,_=reconstruct(tokens)
            history=[(jq.name_to_square(t[:2]),jq.name_to_square(t[2:4])) for t in tokens]
            policy,move,root=engine.search(state,belief,history,temperature=1e-6,simulations=int(simulations) if simulations else 1000000,return_root=True)
            print('info string '+json.dumps({'simulations':root.visits,'elapsed_ms':(time.perf_counter()-started)*1000,'device':a.device,'phase_experts':metadata['phase_experts'],'visit_mass':sum(policy)}),flush=True)
            print('bestmove '+(jq.format_move_uci(move) if move else '0000'),flush=True)
        elif text=='quit':break

if __name__=='__main__':main()
