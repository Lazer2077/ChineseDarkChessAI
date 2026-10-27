"""Exact-node initial-position decoder integration, including legacy control."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import fresh_game,sha,save_json
from research.policy_value import AZ
from research.native_opponent_policy import FixedDepthOpponent


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=ROOT/a.out
    if out.exists():raise ValueError('Refuse to overwrite actor smoke')
    state,belief,_=fresh_game(8320001);legal={AZ.jq.format_move_uci(m) for m in state.legal_moves()}
    root=ROOT/'runs/spatial-policy-integration-01';rows=[]
    cases=[('dense-legacy','adapters/neural_policy_gpu_uci.sh',root/'dense2550'),
           ('dense-new-loader','adapters/spatial_policy_gpu_uci.sh',root/'dense2550'),
           ('relative-new-loader','adapters/spatial_policy_gpu_uci.sh',root/'relative50')]
    for label,adapter,folder in cases:
        with FixedDepthOpponent(ROOT/adapter,folder/'best.npz',depth=1,timeout=120.) as engine:
            engine._send('position startpos');engine._send('go nodes 8');lines=engine._until('bestmove ')
            move=lines[-1].split()[1];stats=[json.loads(x[len('info string '):]) for x in lines if x.startswith('info string {')]
            if len(stats)!=1 or stats[0]['simulations']!=8 or move not in legal:raise ValueError('Actor geometry/completed visits differ')
            rows.append({'label':label,'adapter_sha256':sha(ROOT/adapter),'model_sha256':sha(folder/'best.npz'),
                         'move':move,'completed_simulations':stats[0]['simulations'],'opening_legal':True})
    if rows[0]['move']!=rows[1]['move']:raise ValueError('Dense legacy/new-loader behavior differs')
    save_json(out,{'schema':1,'status':'completed','scope':'GPU opening-only eight-completed-simulation smoke with tiny one-epoch TRAIN-subset models. Dense old/new loaders agree; relative policy produces a legal action. No full game, timed-budget, generalization or strength evidence.','cases':rows})
    print(json.dumps({'status':'completed','dense_legacy_matches_new_loader':True,'cases':len(rows)}),flush=True)


if __name__=='__main__':main()
