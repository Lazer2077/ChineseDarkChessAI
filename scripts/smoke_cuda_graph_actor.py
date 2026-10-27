"""Preload at isready and compare complete UCI opening decisions at fixed work."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.native_opponent_policy import FixedDepthOpponent
from research.neural_replay import fresh_game, sha, save_json
from research.policy_value import AZ


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise ValueError('Refuse to overwrite actor smoke')
    state, _, _ = fresh_game(8320001)
    legal = {AZ.jq.format_move_uci(m) for m in state.legal_moves()}
    model = ROOT / 'runs/neural-policy-phase-02/shared/best.npz'
    cases = []
    for adapter in ['adapters/spatial_policy_gpu_uci.sh', 'adapters/neural_policy_graph_gpu_uci.sh']:
        started = time.perf_counter()
        with FixedDepthOpponent(ROOT / adapter, model, depth=1, timeout=120.) as engine:
            engine._send('setoption name LeafBatch value 32')
            engine._send('isready')
            engine._until('readyok')
            preload_seconds = time.perf_counter() - started
            engine._send('position startpos')
            engine._send('go nodes 8')
            lines = engine._until('bestmove ')
            move = lines[-1].split()[1]
            stats = [json.loads(line[len('info string '):]) for line in lines if line.startswith('info string {')]
            if len(stats) != 1 or stats[0]['simulations'] != 8 or move not in legal:
                raise ValueError('Wrong node count or illegal action')
            cases.append({'adapter': adapter, 'adapter_sha256': sha(ROOT / adapter),
                          'move': move, 'simulations': stats[0]['simulations'],
                          'preload_with_repeated_ready_seconds': preload_seconds,
                          'opening_legal': True})
    if cases[0]['move'] != cases[1]['move']:
        raise ValueError('Graph actor fixed-work decision differs')
    save_json(out, {'schema': 1, 'status': 'completed', 'cases': cases,
                    'model_sha256': sha(model), 'script_sha256': sha(__file__),
                    'scope': 'Opening-only complete UCI control,32leaf batches,8completed simulations each. Both ready paths finish before go. No full-game, actual-clock strength or startup-throughput claim.'})
    print(json.dumps({'status': 'completed', 'move': cases[0]['move']}), flush=True)


if __name__ == '__main__':
    main()
