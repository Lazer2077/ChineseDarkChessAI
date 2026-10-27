"""Full-size GPU memory/update/interchange/search preflight; no fitted strength claim."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.policy_value import REC, encode_records, action_indices, load_npz, export_npz
from research.spatial_policy_value import make_model, load_model
from research.neural_replay import fresh_game, sha, save_json
from research.neural_search import ObservationPUCT


def latency(net, raw, batch):
    def run():
        x = encode_records(raw[:batch])
        with torch.inference_mode():
            policy, value = net(x)
            return torch.softmax(policy.float(), 1).cpu().numpy(), value.float().cpu().numpy()
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / 10


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_text())
    out = ROOT / plan['out']
    scratch = ROOT / plan['scratch']
    if out.exists() or scratch.exists():
        raise ValueError('Refuse to overwrite full-size preflight')
    for name, expected in plan['sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen scale preflight input changed: ' + name)
    scratch.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(plan['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    initial_digest = sha(ROOT / plan['initial'])
    net = make_model('relative50', plan['channels'], plan['blocks'])
    load_npz(net, ROOT / plan['initial'])
    net.to('cuda').train()
    records = np.fromfile(ROOT / plan['train_data'], dtype=REC, count=plan['batch'])
    if len(records) != plan['batch']:
        raise ValueError('Full-batch preflight data missing')
    labels = torch.from_numpy(action_indices(records)).to('cuda')
    target = torch.from_numpy((.7*np.tanh(records['score'].astype(np.float32)/860)
                               + .3*records['result'].astype(np.float32)).copy()).to('cuda')
    raw = torch.from_numpy(records.view(np.uint8).reshape(-1,112)).to('cuda')
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0003, weight_decay=.0001)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda', dtype=dtype):
        policy, value = net(encode_records(raw))
    loss = F.cross_entropy(policy.float(), labels) + F.mse_loss(value.float(), target)
    if not torch.isfinite(loss) or policy.shape != (plan['batch'],2550):
        raise ValueError('Invalid full-size training outputs')
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
    if not torch.isfinite(norm):
        raise ValueError('Invalid full-size gradient')
    scaler.step(optimizer)
    scaler.update()
    peak = torch.cuda.max_memory_allocated()
    file = scratch / 'one-update.npz'
    export_npz(net, file)
    reloaded = make_model('relative50', plan['channels'], plan['blocks'])
    load_npz(reloaded, file)
    if not all(torch.equal(v.cpu(), reloaded.state_dict()[k]) for k,v in net.state_dict().items()):
        raise ValueError('Full-size safe array roundtrip differs')
    del optimizer, reloaded
    net.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    parent, _ = load_model(ROOT / plan['parent'], 'cuda')
    costs = {label: {str(b):latency(model,raw,b) for b in [1,8,32]}
             for label,model in [('parent128x15',parent),('scale256x20',net)]}
    state, belief, _ = fresh_game(plan['opening_seed'])
    engine = ObservationPUCT(net, 'cuda', seed=plan['opening_seed'], leaf_batch=32)
    _, move, root = engine.search(state,belief,[],simulations=8,temperature=0.,return_root=True)
    if move not in state.legal_moves() or root.visits != 8:
        raise ValueError('Full-size opening search integration failed')
    if sha(ROOT / plan['initial']) != initial_digest:
        raise ValueError('Preflight changed the planned cold initial model')
    save_json(out, {'schema':1,'status':'completed','plan_sha256':sha(path),
        'channels':plan['channels'],'blocks':plan['blocks'],'parameters':sum(p.numel() for p in net.parameters()),
        'batch':plan['batch'],'precision':str(dtype),'updates':1,
        'finite_loss':float(loss.detach()),'finite_preclip_gradient_norm':float(norm),
        'peak_allocated_bytes':peak,'safe_numeric_roundtrip_exact':True,
        'planned_initial_unchanged':True,'completed_opening_simulations':root.visits,
        'opening_move_legal':True,'encode_forward_cpu_return_mean_ms':costs,
        'scope':'One full-size GPU update on first512TRAIN rows, numeric interchange and bounded opening search. Timing uses10local repeats with eagerFP32 inference; parent trained versus scale one-update weights is cost only. No checkpoint selection, generalization, whole-game timing guarantee or playing-strength evidence.'})


if __name__ == '__main__':
    main()
