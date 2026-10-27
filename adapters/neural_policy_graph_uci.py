"""Original observer PUCT with preloaded, frozen FP32 forward graphs."""
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'adapters'))
import torch
import neural_policy_uci as protocol
from research.cuda_graph_inference import GraphPolicyValue
from research.policy_value import AZ
from research.spatial_policy_value import load_model


def load_graph_model(directory, device='cuda'):
    if device != 'cuda':
        raise ValueError('This profile requires CUDA')
    net, metadata = load_model(directory, device)
    # Protocol isready loads all shapes before actual per-move clocks start.
    with torch.inference_mode():
        x = torch.zeros((1, AZ.STATE_CHANNELS, 10, 9), device=device)
        net = GraphPolicyValue(net, [x.repeat(batch, 1, 1, 1) for batch in range(1, 33)])
    return net, {**metadata, 'execution': 'cuda-graph-fp32', 'preloaded_graph_batches': 32}


if __name__ == '__main__':
    protocol.load_model = load_graph_model
    hashes = [hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in
              ['research/cuda_graph_inference.py', 'research/spatial_policy_value.py',
               'adapters/neural_policy_graph_uci.py']]
    protocol.TIMED_SEARCH_SHA256 = hashlib.sha256(
        (protocol.TIMED_SEARCH_SHA256 + ''.join(hashes)).encode()).hexdigest()
    protocol.main()
