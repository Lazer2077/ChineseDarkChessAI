"""FP32 forward capture for a frozen, single-threaded inference model.

Only neural arithmetic is captured. Observer encoding, legality, information
sets, rule histories, chance sampling and search remain outside these graphs.
"""
import threading

import torch


class GraphPolicyValue(torch.nn.Module):
    def __init__(self, model, samples, warmup=3):
        super().__init__()
        self.model = model.eval()
        self.owner_thread = threading.get_ident()
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.eval()
        if not samples:
            raise ValueError('Require declared capture shapes')
        with torch.inference_mode():
            for sample in samples:
                if sample.device.type != 'cuda' or sample.dtype != torch.float32:
                    raise ValueError('Require CUDA FP32 observer tensors')
                key = (tuple(sample.shape), sample.device.index, sample.dtype)
                if key in self.graphs:
                    raise ValueError('Duplicate capture shape')
                static = sample.detach().clone()
                stream = torch.cuda.Stream(device=sample.device)
                stream.wait_stream(torch.cuda.current_stream(sample.device))
                with torch.cuda.stream(stream):
                    for _ in range(warmup):
                        self.model(static)
                torch.cuda.current_stream(sample.device).wait_stream(stream)
                torch.cuda.synchronize(sample.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self.pool):
                    outputs = self.model(static)
                if not isinstance(outputs, tuple) or len(outputs) != 2:
                    raise ValueError('Require a policy/value output pair')
                self.graphs[key] = (static, graph, outputs)
        torch.cuda.synchronize()

    def train(self, mode=True):
        if mode:
            raise ValueError('Captured inference model cannot train or change weights')
        return super().train(False)

    def forward(self, x):
        if threading.get_ident() != self.owner_thread or torch.is_grad_enabled():
            raise ValueError('Require original single inference thread without gradients')
        key = (tuple(x.shape), x.device.index, x.dtype)
        if key not in self.graphs:
            # No capture or shape padding on a timed search path.
            return self.model(x)
        static, graph, outputs = self.graphs[key]
        static.copy_(x)
        graph.replay()
        # Preserve normal tensor lifetime semantics across later graph calls.
        return tuple(output.clone() for output in outputs)
