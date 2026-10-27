"""CPU cross-matrix: CE = target entropy + KL on reused selection trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
from train_neural_selfplay import load_split
from research.neural_search import load_model
from research.policy_value import encode_records
from research.neural_replay import save_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True); a = p.parse_args()
    path = Path(a.plan); plan = json.loads(path.read_text()); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    def verify():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise ValueError('Plan changed')
        for name, expected in plan['sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected: raise ValueError('Frozen input changed: '+name)
    verify(); out = ROOT/plan['out']; out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    report = {'schema': 1, 'status': 'running', 'plan_sha256': digest, 'scope': plan['scope'], 'datasets': [], 'matrix': []}
    save_json(out/'report.json', report)
    for dataset in plan['datasets']:
        folder = ROOT/dataset['episodes']; manifest = json.loads((folder/'manifest.json').read_text())
        if manifest['status'] != 'completed': raise ValueError('Require completed fixed generation')
        raw, pi, z, meta = load_split(folder, dataset['indices'], manifest)
        if pi.dtype != np.float64: raise ValueError('Preserve original float64 normalized visit targets')
        entropy_each = -(pi*np.log(np.maximum(pi, np.finfo(np.float64).tiny))).sum(1)
        entropy = float(entropy_each.mean())
        report['datasets'].append({'label': dataset['label'], 'positions': len(raw), 'episodes': meta,
            'target_entropy_nats': entropy, 'mean_target_max_probability': float(pi.max(1).mean()),
            'mean_visited_actions': float((pi > 0).sum(1).mean()),
            'fully_public_positions': int(((raw[:,:90]&15) != 8).all(1).sum()),
            'mean_actor_terminal_return': float(z.mean())})
        for model in plan['models']:
            net, best = load_model(ROOT/model['directory'], 'cpu'); totals = np.zeros(3, dtype=np.float64)
            with torch.inference_mode():
                for start in range(0, len(raw), plan['batch']):
                    sl = slice(start, start+plan['batch'])
                    logits, value = net(encode_records(torch.from_numpy(raw[sl])))
                    target = torch.from_numpy(pi[sl]); terminal = torch.from_numpy(z[sl])
                    ce = -(target*F.log_softmax(logits.float(), dim=1)).sum(1)
                    mse = (value.float()-terminal)**2
                    totals += [float(ce.sum()), float(mse.sum()), len(ce)]
            ce, mse = float(totals[0]/totals[2]), float(totals[1]/totals[2])
            if ce-entropy < -1e-6: raise ValueError('Invalid negative policy KL')
            row = {'dataset': dataset['label'], 'model': model['label'], 'policy_ce': ce,
                   'target_entropy_nats': entropy, 'policy_kl_nats': ce-entropy, 'terminal_value_mse': mse,
                   'network_sha256': best['npz_sha256']}
            if model.get('own_dataset') == dataset['label']:
                expected = best['validation']
                if abs(ce-expected['search_policy_ce']) > 3e-4 or abs(mse-expected['terminal_value_mse']) > 3e-4:
                    raise ValueError('CPU evaluation does not reproduce archived selection losses')
                row['own_selection_metrics_reproduced'] = True
            report['matrix'].append(row); verify(); save_json(out/'report.json', report)
            print(json.dumps(row), flush=True)
            del net
    verify(); report['status'] = 'completed'; save_json(out/'report.json', report)


if __name__ == '__main__': main()
