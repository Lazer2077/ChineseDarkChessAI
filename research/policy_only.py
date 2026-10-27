"""Greedy legal-normalized prior ablation: one inference, zero tree simulations."""
from research.neural_search import ObservationPUCT
from research.policy_value import AZ


class GreedyPolicy(ObservationPUCT):
    def search(self, root_state, belief, history, *, return_root=False, **kwargs):
        if kwargs.get('add_dirichlet', False): raise ValueError('Greedy policy has no exploration noise')
        info = belief.info_state(root_state, root_state.side)
        priors, _ = self._evaluate(info, history)
        root = AZ.PUCTNode(1., root_state.side)
        # Reuse the original legal normalization/tie order; no chance expansion,
        # determinizations, selections, leaf rollouts or visit backups occur.
        root.expand(root_state.legal_moves(), priors)
        policy = [0.]*AZ.MOVE_SIZE
        for move, child in root.children.items(): policy[AZ.move_to_index(move)] = child.prior
        move = max(root.children, key=lambda m: root.children[m].prior) if root.children else None
        return (policy, move, root) if return_root else (policy, move)
