"""Check complete fixed-work PUCT fingerprints before graph inference deployment."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from research.cuda_graph_inference import GraphPolicyValue
from research.neural_replay import fresh_game, load_episode, sha, save_json
from research.neural_search import ObservationPUCT
from research.policy_value import AZ, encode_records, records_from_info
from research.spatial_policy_value import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    digest = sha(args.plan)
    out = ROOT / plan['out']
    if out.exists():
        raise ValueError('Refuse to overwrite fixed-work graph control')

    def verify():
        if sha(args.plan) != digest:
            raise ValueError('Graph search plan changed')
        for name, expected in plan['sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Frozen graph search input changed: ' + name)

    verify()
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases = []
    for case in plan['cases']:
        data, meta = load_episode(ROOT / plan['episodes'] / f"game-{case['game']:06d}.npz")
        state, belief, rng = fresh_game(meta['seed'])
        history = []
        for move in data['moves'][:case['ply']]:
            move = tuple(map(int, move))
            AZ.im.apply_move_and_update_belief(state, belief, move, rng=rng)
            history.append(move)
        cases.append((state, belief, history))
    report = {'schema': 1, 'status': 'running', 'plan_sha256': digest,
              'scope': plan['scope'], 'samples': [], 'pairs': []}
    save_json(out / 'report.json', report)
    with torch.inference_mode():
        for spec in plan['models']:
            net, _ = load_model(ROOT / spec['directory'], 'cuda')
            state, belief, _ = cases[0]
            x = encode_records(torch.from_numpy(records_from_info(belief.info_state(state, state.side))).cuda())
            started = time.perf_counter()
            graph = GraphPolicyValue(net, [x.repeat(batch, 1, 1, 1) for batch in range(1, 33)])
            capture_seconds = time.perf_counter() - started
            for model in [net, graph]:
                engine = ObservationPUCT(model, 'cuda', seed=0, leaf_batch=32)
                engine.search(*cases[0], temperature=0, add_dirichlet=False, simulations=32)
            for case_index, (state, belief, history) in enumerate(cases):
                for repeat in range(plan['repeats']):
                    current = {}
                    for mode in (['eager', 'graph'] if repeat % 2 == 0 else ['graph', 'eager']):
                        engine = ObservationPUCT(net if mode == 'eager' else graph, 'cuda',
                                                 seed=plan['seed'] + case_index, leaf_batch=32)
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        policy, move, root = engine.search(state, belief, history, temperature=0,
                            simulations=plan['simulations'], add_dirichlet=False, return_root=True)
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - started
                        if root.visits != plan['simulations'] or sum(c.visits for c in root.children.values()) != root.visits:
                            raise ValueError('Incomplete root backups')
                        fingerprint = {'move': list(move), 'policy': policy, 'root_visits': root.visits,
                            'root_value_sum': root.value_sum,
                            'children': [[list(m), c.visits, c.value_sum, c.prior] for m, c in root.children.items()]}
                        sample = {'model': spec['id'], 'case': case_index, 'repeat': repeat,
                                  'mode': mode, 'seconds': seconds, 'fingerprint': fingerprint}
                        report['samples'].append(sample)
                        current[mode] = sample
                    same = current['eager']['fingerprint'] == current['graph']['fingerprint']
                    report['pairs'].append({'model': spec['id'], 'case': case_index, 'repeat': repeat,
                        'exact_root_statistics_match': same,
                        'eager_over_graph_seconds': current['eager']['seconds'] / current['graph']['seconds']})
                    save_json(out / 'report.json', report)
                    if not same:
                        raise ValueError('Graph changed complete fixed-work root fingerprint')
            print(json.dumps({'completed_model': spec['id'], 'capture_seconds': capture_seconds}), flush=True)
            del graph, net
            torch.cuda.empty_cache()
    report['median_ratios'] = [{'model': spec['id'], 'case': index,
        'phase': case['phase'], 'eager_over_graph_seconds': statistics.median(r['eager_over_graph_seconds']
            for r in report['pairs'] if (r['model'], r['case']) == (spec['id'], index))}
        for spec in plan['models'] for index, case in enumerate(plan['cases'])]
    report.update(status='completed', all_root_statistics_match=True, limitations=plan['limitations'])
    verify()
    save_json(out / 'report.json', report)
    save_json(ROOT / plan['archive'], report)
    print(json.dumps({'status': 'completed', 'median_ratios': report['median_ratios']}), flush=True)


if __name__ == '__main__':
    main()
