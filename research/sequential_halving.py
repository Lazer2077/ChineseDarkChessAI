"""Fixed-work Gumbel root halving with inherited information-set PUCT inside.

Root-only research ablation, not full Gumbel AlphaZero: no completed-Q training
target or deterministic interior policy update. No policy-improvement theorem
is claimed for approximate neural/particle Q values. No timed acting interface.
"""
import hashlib
import inspect
import math
import random
import textwrap
from research.neural_search import ObservationPUCT
from research.policy_value import AZ


class RootHalving:
    def __init__(self, budget, candidates=16, seed=0, noise_scale=0., c_visit=50.):
        if type(budget) is not int or budget < 1 or type(candidates) is not int or candidates < 1:
            raise ValueError('Require positive integer budget and candidate count')
        if not math.isfinite(noise_scale) or noise_scale < 0 or not math.isfinite(c_visit) or c_visit < 0:
            raise ValueError('Invalid root score parameters')
        self.budget, self.candidates = budget, candidates
        self.rng, self.noise_scale, self.c_visit = random.Random(seed), noise_scale, c_visit
        self.active = None; self.assigned = 0; self.trace = []

    def _initialize(self, root):
        self.log_prior_noise = {}
        for move, child in root.children.items():
            noise = -math.log(-math.log(max(self.rng.random(), 1e-300))) if self.noise_scale else 0.
            self.log_prior_noise[move] = math.log(max(child.prior, 1e-300))+self.noise_scale*noise
        self.active = sorted(root.children, key=self.log_prior_noise.get, reverse=True)[:min(self.candidates, self.budget)]
        self._start_round()

    def _start_round(self):
        remaining = self.budget-self.assigned
        rounds = max(1, math.ceil(math.log2(len(self.active))))
        per_action = max(1, remaining//rounds//len(self.active))
        self.round_cost = min(remaining, per_action*len(self.active))
        self.round_assigned = 0
        self.trace.append({'active': list(self.active), 'start': self.assigned, 'cost': self.round_cost})

    def score(self, root, move):
        # Child edge Q is the next player's value. Map root Q[-1,1] to[0,1];
        # the common +0.5 term cancels in every action comparison.
        scale = (self.c_visit+max(c.visits for c in root.children.values()))*.5
        return self.log_prior_noise[move]-scale*root.children[move].q_value()

    def _advance_completed_round(self, root):
        if self.round_assigned == self.round_cost:
            if root.visits != self.assigned:
                raise ValueError('Never halve while virtual credits are outstanding')
            self.active = sorted(self.active, key=lambda m: self.score(root, m), reverse=True)[:max(1, (len(self.active)+1)//2)]
            if self.assigned < self.budget: self._start_round()

    def capacity(self, root):
        if not root.children: return 1
        if self.active is None: self._initialize(root)
        else: self._advance_completed_round(root)
        capacity = self.round_cost-self.round_assigned
        if capacity <= 0: raise ValueError('Fixed budget exhausted')
        return capacity

    def choose(self, root):
        if self.active is None or self.round_assigned >= self.round_cost:
            raise ValueError('Start a completed-batch round before selection')
        move = self.active[self.round_assigned % len(self.active)]
        self.round_assigned += 1; self.assigned += 1
        return move, root.children[move]

    def recommend(self, root):
        if not root.children: return None
        if self.assigned != self.budget or root.visits != self.budget:
            raise ValueError('Incomplete fixed-work search')
        self._advance_completed_round(root)
        # Handles rounding/incomplete halving: use the most visited root set,
        # then the same noise/prior/scaled-Q score, not insertion-order ties.
        most = max(c.visits for c in root.children.values())
        return max((m for m, c in root.children.items() if c.visits == most), key=lambda m: self.score(root, m))


class HalvingPUCT(ObservationPUCT):
    def search(self, root_state, belief, history, *, simulations, gumbel_seed, noise_scale=0., candidates=16,
               return_root=False, temperature=0., add_dirichlet=False):
        if self.deadline is not None or add_dirichlet or temperature > 1e-3:
            raise ValueError('Fixed work, explicit Gumbel seed and recommendation only; no timed/Dirichlet/visit-sampling profile')
        self.halving = RootHalving(simulations, candidates, gumbel_seed, noise_scale)
        policy, _, root = self._fixed_search(root_state, belief, history, simulations=simulations,
                                            temperature=0., add_dirichlet=False, return_root=True)
        move = self.halving.recommend(root)
        return (policy, move, root) if return_root else (policy, move)


def _install():
    original = textwrap.dedent(inspect.getsource(AZ.NeuralMCTS.search))
    batch = '        n_batch = min(batch_size, sims - completed)\n'
    select = '                move, child = node.select_child(self.config.c_puct)\n'
    if original.count(batch) != 1 or original.count(select) != 1:
        raise ValueError('Pinned root search sites changed')
    source = original.replace(batch, batch+'        n_batch = min(n_batch, self.halving.capacity(root))\n')
    source = source.replace(select, '                move, child = (self.halving.choose(root) if node is root else node.select_child(self.config.c_puct))\n')
    namespace = dict(AZ.__dict__)
    exec(compile(source, '<cdca-fixed-root-halving>', 'exec'), namespace)
    HalvingPUCT._fixed_search = namespace['search']
    return hashlib.sha256(source.encode()).hexdigest()


SEARCH_SHA256 = _install()
