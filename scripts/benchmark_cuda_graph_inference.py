"""Compare eager and captured FP32 on frozen existing observer inputs."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from research.cuda_graph_inference import GraphPolicyValue
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
        raise ValueError('Refuse to overwrite graph benchmark')

    def verify():
        if sha(args.plan) != digest:
            raise ValueError('Graph benchmark plan changed')
        for name, expected in plan['sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Frozen graph input changed: ' + name)

    verify()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = json.loads((ROOT / plan['sample_report']).read_text())
    records = np.memmap(ROOT / plan['train_replay'], dtype=REC, mode='r')
    raw = torch.from_numpy(np.array(records[source['sample_indices']]).view(np.uint8).reshape(-1, 112)).cuda()
    encoded = encode_records(raw)
    report = {'schema': 1, 'status': 'running', 'plan_sha256': digest,
              'scope': plan['scope'], 'samples': [], 'checks': [],
              'captured_arithmetic': 'FP32 with original unchanged TF32 settings; no autocast, model training or observation changes.'}
    with torch.inference_mode():
        for spec in plan['models']:
            if 'directory' in spec:
                net, _ = load_model(ROOT / spec['directory'], 'cuda')
            else:
                net = make_model(spec['policy_head'], spec['channels'], spec['blocks'])
                load_npz(net, ROOT / spec['npz'])
                net.cuda().eval()
            before = {k: v.clone() for k, v in net.state_dict().items()}
            started = time.perf_counter()
            graphed = GraphPolicyValue(net, [encoded[:b] for b in plan['batch_sizes']])
            capture_seconds = time.perf_counter() - started
            count = 0
            retained = None
            for batch in plan['batch_sizes']:
                for start in range(0, len(encoded) - batch + 1, batch):
                    x = encoded[start:start + batch]
                    reference = net(x)
                    got = graphed(x)
                    if not all(torch.equal(a, b) for a, b in zip(reference, got)):
                        raise ValueError('Captured FP32 differs from eager outputs')
                    if retained is not None and not all(torch.equal(a, b) for a, b in zip(*retained)):
                        raise ValueError('Later graph replay overwrote previously returned tensors')
                    retained = (got, tuple(y.clone() for y in got))
                    count += batch
            # A noncaptured shape must use exactly the original eager model.
            fallback = encoded[:3]
            if not all(torch.equal(a, b) for a, b in zip(net(fallback), graphed(fallback))):
                raise ValueError('Fallback shape differs')
            if any(not torch.equal(before[k], v) for k, v in net.state_dict().items()):
                raise ValueError('Graph warmup/replay changed frozen weights or buffers')
            report['checks'].append({'model': spec['id'], 'positions_checked_across_shapes': count,
                'bitwise_policy_value_match': True, 'earlier_output_lifetimes_preserved': True,
                'uncaptured_shape_eager_match': True, 'weights_and_buffers_unchanged': True,
                'capture_seconds_excluded_from_timings': capture_seconds})
            for batch in plan['batch_sizes']:
                for path in ['forward_softmax', 'encode_forward_cpu_return']:
                    for repeat in range(plan['repeats']):
                        for mode in (['eager', 'graph'] if repeat % 2 == 0 else ['graph', 'eager']):
                            model = net if mode == 'eager' else graphed

                            def call():
                                x = encoded[:batch] if path == 'forward_softmax' else encode_records(raw[:batch])
                                pp, vv = model(x)
                                pp, vv = torch.softmax(pp.float(), dim=1), vv.float()
                                if path != 'forward_softmax':
                                    return pp.cpu().numpy(), vv.cpu().tolist()
                                return pp, vv

                            for _ in range(plan['warmup_calls']):
                                call()
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            for _ in range(plan['timed_calls']):
                                call()
                            torch.cuda.synchronize()
                            report['samples'].append({'model': spec['id'], 'batch': batch, 'path': path,
                                'repeat': repeat, 'mode': mode,
                                'milliseconds_per_call': (time.perf_counter() - started) * 1000 / plan['timed_calls']})
            del graphed, net
            torch.cuda.empty_cache()
            print(json.dumps({'completed_model': spec['id']}), flush=True)
    summary = []
    for spec in plan['models']:
        for batch in plan['batch_sizes']:
            for path in ['forward_softmax', 'encode_forward_cpu_return']:
                times = {mode: statistics.median(r['milliseconds_per_call'] for r in report['samples']
                    if (r['model'], r['batch'], r['path'], r['mode']) == (spec['id'], batch, path, mode))
                    for mode in ['eager', 'graph']}
                summary.append({'model': spec['id'], 'batch': batch, 'path': path,
                                'median_ms': times, 'eager_over_graph': times['eager'] / times['graph']})
    verify()
    report.update(status='completed', summary=summary, limitations=plan['limitations'])
    save_json(out, report)
    print(json.dumps({'status': 'completed', 'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
