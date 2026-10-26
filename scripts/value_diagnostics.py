"""Describe value errors by observable hidden-piece count; not a strength test."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--model', action='append', required=True, help='label:checkpoint.pt')
p.add_argument('--out', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
source = root / 'external/devinokeefe-jieqi/engine/train/train.py'
spec = importlib.util.spec_from_file_location('diagnostic_backbone', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
data = Path(a.data)
if data.suffix != '.bin' or data.stat().st_size % 112:
    raise SystemExit('Require a complete record file, never a .part file')
if Path(a.out).exists():
    raise SystemExit('Refuse to overwrite diagnostics')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
raw, total = m.load_records([str(data)])
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
records = torch.from_numpy(raw).to(dev)
board, stm, urn, scores, results, _ = m.fields(records)
truth = (results.float() + 1) / 2
search = torch.sigmoid(scores.float() / 430)
target = .7 * search + .3 * truth
hidden = ((board & 15) == 8).sum(1)
strata = {'all': torch.ones(len(records), dtype=torch.bool, device=dev),
          'hidden_0': hidden == 0, 'hidden_1_4': (hidden >= 1) & (hidden <= 4),
          'hidden_5_14': (hidden >= 5) & (hidden <= 14), 'hidden_15_30': hidden >= 15}
report = {'schema': 1, 'dataset_sha256': sha(data), 'raw_records': total,
          'retained_records': len(records), 'score_scale': 430, 'lambda': .7,
          'trainer_sha256': sha(source), 'models': {},
          'scope': 'Descriptive diagnostics on checkpoint-selection validation data; positions within games are correlated. No independent test or strength confidence interval. Hidden-count strata use actor-visible information only.'}
for item in a.model:
    label, filename = item.split(':', 1)
    if label in report['models']:
        raise SystemExit('Duplicate model label')
    checkpoint = Path(filename)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    features, ft = state['ft.weight'].shape
    kb, rem = divmod(features - 1 - 30, 1440)
    if rem or kb not in (1, 3, 9):
        raise SystemExit('Unsupported feature shape')
    m.set_king_buckets(kb)
    net = m.Net(ft - 1, state['l1.weight'].shape[0], state['l2.weight'].shape[0]).to(dev)
    net.load_state_dict(state)
    net.eval()
    feat = m.Featurizer(dev)
    with torch.no_grad():
        predictions = []
        for start in range(0, len(records), 4096):
            b = slice(start, start + 4096)
            us, them = feat(board[b], stm[b], urn[b])
            predictions.append(torch.sigmoid(net(us, them) * net.out_scale / 430))
        pred = torch.cat(predictions)
        if not torch.isfinite(pred).all():
            raise SystemExit('Nonfinite predictions')
        groups = {}
        for name, keep in strata.items():
            n = int(keep.sum())
            if not n:
                groups[name] = {'positions': 0}
                continue
            prob = pred[keep].clamp(1e-6, 1 - 1e-6)
            y = truth[keep]
            groups[name] = {'positions': n,
                'mixed_target_mse': float(((pred[keep] - target[keep]) ** 2).mean()),
                'search_target_mse': float(((pred[keep] - search[keep]) ** 2).mean()),
                'outcome_brier': float(((pred[keep] - y) ** 2).mean()),
                'outcome_logloss': float((-(y * prob.log() + (1 - y) * (1 - prob).log())).mean()),
                'mean_prediction': float(pred[keep].mean()), 'mean_outcome': float(y.mean())}
    report['models'][label] = {'checkpoint_sha256': sha(checkpoint), 'parameters': sum(t.numel() for t in net.parameters()), 'king_buckets': kb, 'strata': groups}
    del net
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
