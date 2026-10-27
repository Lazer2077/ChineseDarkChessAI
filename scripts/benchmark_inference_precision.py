"""Measure forward-only and full evaluation cost; no policy or search is changed."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from research.neural_replay import sha, save_json
from research.policy_value import REC, encode_records, load_npz
from research.spatial_policy_value import load_model, make_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    digest = sha(args.plan)
    out = ROOT / plan['out']
    if out.exists():
        raise ValueError('Refuse to overwrite precision benchmark')

    def verify():
        if sha(args.plan) != digest:
            raise ValueError('Benchmark plan changed')
        for name, expected in plan['sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Frozen benchmark input changed: ' + name)

    verify()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_bf16_supported():
        raise ValueError('Require native BF16 support for this fixed study')
    records = np.memmap(ROOT / plan['train_replay'], dtype=REC, mode='r')
    selected = {phase: [] for phase in range(4)}
    for start in range(0, len(records), 4096):
        chunk = records[start:start + 4096]
        dark = ((chunk['board'] & 15) == 8).sum(1)
        bins = np.where(dark == 0, 0, np.where(dark <= 4, 1, np.where(dark <= 14, 2, 3)))
        for phase in selected:
            need = plan['positions_per_phase'] - len(selected[phase])
            selected[phase].extend((start + np.flatnonzero(bins == phase)[:need]).tolist())
        if all(len(v) == plan['positions_per_phase'] for v in selected.values()):
            break
    if not all(len(v) == plan['positions_per_phase'] for v in selected.values()):
        raise ValueError('Insufficient prespecified phase exposure')
    indices = np.asarray([i for values in selected.values() for i in values], dtype=np.int64)
    raw = np.array(records[indices]).view(np.uint8).reshape(-1, 112)
    # Interleave phases so even the smallest microbenchmark has a declared input.
    order = np.arange(len(raw)).reshape(4, -1).T.reshape(-1)
    raw = raw[order]
    indices = indices[order]
    tensor = torch.from_numpy(raw).cuda()
    encoded = encode_records(tensor)
    report = {'schema': 1, 'status': 'running', 'plan_sha256': digest,
              'scope': plan['scope'], 'sample_indices': indices.tolist(),
              'sample_live_dark_counts': ((records[indices]['board'] & 15) == 8).sum(1).tolist(),
              'torch_version': torch.__version__, 'cpu_threads': 1,
              'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
              'tf32_cudnn': torch.backends.cudnn.allow_tf32,
              'samples': [], 'numeric_comparisons': [],
              'precision_context': 'Autocast entered separately for every forward, FP32 stored parameters; softmax and returned values explicitly FP32. Model eval/inference mode; no weight updates.'}

    def context(precision):
        return torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()

    def outputs(net, x, precision):
        with context(precision):
            policy, value = net(x)
        return torch.softmax(policy.float(), dim=1), value.float()

    with torch.inference_mode():
        for spec in plan['models']:
            verify()
            if 'directory' in spec:
                net, _ = load_model(ROOT / spec['directory'], 'cuda')
            else:
                net = make_model(spec['policy_head'], spec['channels'], spec['blocks'])
                load_npz(net, ROOT / spec['npz'])
                net.cuda().eval()
            reference_p, reference_v = outputs(net, encoded, 'fp32')
            approx_p, approx_v = outputs(net, encoded, 'bf16')
            if not all(torch.isfinite(x).all() for x in [reference_p, reference_v, approx_p, approx_v]):
                raise ValueError('Nonfinite precision output')
            report['numeric_comparisons'].append({
                'model': spec['id'], 'trained': spec['trained'],
                'value_mean_absolute_difference': (reference_v - approx_v).abs().mean().item(),
                'value_max_absolute_difference': (reference_v - approx_v).abs().max().item(),
                'mean_policy_total_variation': (reference_p - approx_p).abs().sum(1).mean().item() / 2,
                'full_geometry_top1_disagreements': (reference_p.argmax(1) != approx_p.argmax(1)).sum().item(),
                'positions': len(encoded), 'all_finite': True})
            for batch in plan['batch_sizes']:
                for path in ['forward_only', 'encode_forward_cpu_return']:
                    for repeat in range(plan['repeats']):
                        for precision in (['fp32', 'bf16'] if repeat % 2 == 0 else ['bf16', 'fp32']):
                            def call():
                                x = encoded[:batch] if path == 'forward_only' else encode_records(tensor[:batch])
                                pp, vv = outputs(net, x, precision)
                                if path != 'forward_only':
                                    return pp.cpu().numpy(), vv.cpu().tolist()
                                return pp, vv
                            for _ in range(plan['warmup_calls']):
                                call()
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            for _ in range(plan['timed_calls']):
                                call()
                            torch.cuda.synchronize()
                            report['samples'].append({'model': spec['id'], 'batch': batch,
                                'path': path, 'repeat': repeat, 'precision': precision,
                                'milliseconds_per_call': (time.perf_counter() - started) * 1000 / plan['timed_calls']})
            del net
            torch.cuda.empty_cache()
            print(json.dumps({'completed_model': spec['id']}), flush=True)
    report['summary'] = []
    for spec in plan['models']:
        for batch in plan['batch_sizes']:
            for path in ['forward_only', 'encode_forward_cpu_return']:
                times = {precision: statistics.median(r['milliseconds_per_call'] for r in report['samples']
                    if (r['model'], r['batch'], r['path'], r['precision']) == (spec['id'], batch, path, precision))
                    for precision in ['fp32', 'bf16']}
                report['summary'].append({'model': spec['id'], 'batch': batch, 'path': path,
                                          'median_ms': times, 'fp32_over_bf16': times['fp32'] / times['bf16']})
    verify()
    report.update(status='completed', limitations=plan['limitations'])
    save_json(out, report)
    print(json.dumps({'status': 'completed', 'summary': report['summary'],
                      'numeric_comparisons': report['numeric_comparisons']}), flush=True)


if __name__ == '__main__':
    main()
