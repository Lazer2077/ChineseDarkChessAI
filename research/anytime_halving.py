"""Timed root-halving prefix followed by PUCT among two surviving actions.

New hybrid ablation, not full Gumbel AlphaZero or a policy-improvement theorem.
The fixed-work prototype stays unchanged. Deadlines interrupt only between
completed leaf batches, so the returned tree has no outstanding virtual loss.
"""
import hashlib
import inspect
import math
import textwrap
import time
from research.neural_search import ObservationPUCT
from research.sequential_halving import RootHalving
from research.policy_value import AZ


class SurvivorHalving(RootHalving):
    def __init__(self, budget=256, candidates=16, c_puct=1.5):
        if budget < 2*candidates: raise ValueError('Prefix must support all minimum halving rounds')
        super().__init__(budget, candidates, seed=0, noise_scale=0.)
        self.tail_started = None; self.c_puct = c_puct

    def _advance_completed_round(self, root):
        if self.tail_started is not None: return
        if self.round_assigned == self.round_cost:
            if root.visits != self.assigned: raise ValueError('Outstanding virtual credits at halving boundary')
            if len(self.active) <= 2:
                self.tail_started = self.assigned
            else:
                super()._advance_completed_round(root)

    def capacity(self, root):
        if not root.children: return 1
        if self.active is None: self._initialize(root)
        else: self._advance_completed_round(root)
        if self.tail_started is not None: return 2**31
        capacity = self.round_cost-self.round_assigned
        if capacity <= 0: raise ValueError('Invalid root prefix allocation')
        return capacity

    def choose(self, root):
        if self.tail_started is None: return super().choose(root)
        # Restricted PUCT uses conditional surviving priors and surviving visit
        # mass, rather than artificially losing prior mass to rejected moves.
        mass = sum(root.children[m].prior for m in self.active)
        total = sum(root.children[m].visits for m in self.active)
        def score(move):
            child = root.children[move]
            prior = child.prior/mass if mass > 0 else 1/len(self.active)
            return -child.q_value()+self.c_puct*prior*math.sqrt(total+1)/(1+child.visits)
        move = max(self.active, key=score); self.assigned += 1
        return move, root.children[move]

    def recommend(self, root):
        if not root.children: return None
        if root.visits < 1 or root.visits != self.assigned:
            raise ValueError('Incomplete root backups')
        self._advance_completed_round(root)
        most = max(c.visits for c in root.children.values())
        return max((m for m,c in root.children.items() if c.visits == most), key=lambda m: self.score(root, m))


class AnytimeHalvingPUCT(ObservationPUCT):
    def __init__(self, net, device='cpu', seed=20261029, leaf_batch=8, prefix_simulations=256):
        super().__init__(net, device, seed, leaf_batch)
        self.prefix_simulations = prefix_simulations

    def search(self, root_state, belief, history, *, simulations=None, temperature=0., add_dirichlet=False, return_root=False):
        if add_dirichlet or temperature > 1e-3: raise ValueError('Deterministic inference-only hybrid profile')
        sims = self.config.mcts_simulations if simulations is None else simulations
        if type(sims) is not int or sims < 1: raise ValueError('Invalid maximum simulation count')
        self.halving = SurvivorHalving(self.prefix_simulations, c_puct=self.config.c_puct)
        policy, _, root = self._timed_search(root_state, belief, history, simulations=sims,
                                             temperature=0., add_dirichlet=False, return_root=True)
        move = self.halving.recommend(root)
        self.last_diagnostics = {'prefix_simulations': self.prefix_simulations, 'completed_simulations': root.visits,
            'survivor_tail_started_after': self.halving.tail_started,
            'remaining_root_candidates': len(self.halving.active or []), 'deadline_enabled': self.deadline is not None}
        return (policy, move, root) if return_root else (policy, move)


def _install():
    source = textwrap.dedent(inspect.getsource(AZ.NeuralMCTS.search))
    loop = '    while completed < sims:\n'
    batch = '        n_batch = min(batch_size, sims - completed)\n'
    select = '                move, child = node.select_child(self.config.c_puct)\n'
    if any(source.count(site) != 1 for site in (loop,batch,select)): raise ValueError('Pinned search sites changed')
    source = source.replace(loop, loop+'        if completed and self.deadline is not None and _cdca_clock() >= self.deadline:\n            break\n')
    source = source.replace(batch, batch+'        n_batch = min(n_batch, self.halving.capacity(root))\n')
    source = source.replace(select, '                move, child = (self.halving.choose(root) if node is root else node.select_child(self.config.c_puct))\n')
    namespace = dict(AZ.__dict__, _cdca_clock=time.perf_counter)
    exec(compile(source, '<cdca-anytime-root-survivors>', 'exec'), namespace)
    AnytimeHalvingPUCT._timed_search = namespace['search']
    return hashlib.sha256(source.encode()).hexdigest()


SEARCH_SHA256 = _install()
