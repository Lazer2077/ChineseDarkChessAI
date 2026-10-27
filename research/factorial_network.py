"""Function-level policy/value attribution without mixing parameters or BN state."""
import torch
from research.neural_search import ObservationPUCT, load_model


class FactorialNetwork(torch.nn.Module):
    def __init__(self, policy_network, value_network):
        super().__init__()
        self.policy_network = policy_network
        self.value_network = value_network

    def forward(self, x):
        # Always execute both complete forwards, including identical checkpoints.
        policy, _ = self.policy_network(x)
        _, value = self.value_network(x)
        return policy, value


class FactorialPUCT(ObservationPUCT):
    def __init__(self, net, device='cpu', seed=20261029, leaf_batch=32,
                 value_directory=None):
        if value_directory is None:
            raise ValueError('Explicit frozen value checkpoint required')
        value_net, self.value_metadata = load_model(value_directory, device)
        combined = FactorialNetwork(net, value_net).to(device).eval()
        super().__init__(combined, device, seed, leaf_batch)
