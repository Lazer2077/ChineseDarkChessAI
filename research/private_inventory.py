"""Exact finite Bayesian inventory filter for a declared count-based policy model.

K includes live own dark pieces and own pieces captured without public reveal.
h counts those private captures; the opponent knows h, the root player does not.
No true hidden identity or opponent private ledger is accepted by this interface.
"""
import itertools
import math
from numbers import Integral


def _counts(values):
    values = tuple(values)
    if not values or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in values):
        raise ValueError('Counts must be nonnegative integers')
    return tuple(int(x) for x in values)


def _logsum(values):
    values = list(values)
    m = max(values, default=-math.inf)
    return m if m == -math.inf else m + math.log(math.fsum(math.exp(x-m) for x in values))


class PrivateInventory:
    def __init__(self, public_counts, captured=0):
        self.public = _counts(public_counts)
        if len(self.public) > 6 or sum(self.public) > 15:
            raise ValueError('Expected at most six types and fifteen dark pieces')
        if isinstance(captured, bool) or not isinstance(captured, Integral) or not 0 <= captured <= sum(self.public):
            raise ValueError('Invalid private capture count')
        self.captured = int(captured)
        self.uniform = True
        logp = {h: sum(math.log(math.comb(k, n)) for k, n in zip(self.public, h))
                for h in itertools.product(*(range(k+1) for k in self.public)) if sum(h) == captured}
        self._commit(self.public, self.captured, logp)

    @property
    def live(self):
        return sum(self.public)-self.captured

    @property
    def log_probabilities(self):
        return dict(self._logp)

    @property
    def probabilities(self):
        return {h: math.exp(v) for h, v in self._logp.items()}

    def _commit(self, public, captured, logp):
        z = _logsum(logp.values())
        if not math.isfinite(z):
            raise ValueError('Impossible observation')
        self.public, self.captured = tuple(public), captured
        self._logp = {h: v-z for h, v in logp.items() if v != -math.inf}

    def condition_action(self, likelihoods):
        """Likelihood of observed action under each h, already policy-normalized.

        Any temperature/epsilon mixture must be applied to each complete masked
        policy BEFORE extracting its observed-action probability.
        """
        if set(likelihoods) != set(self._logp):
            raise ValueError('Likelihood keys must cover current support exactly')
        vals = list(likelihoods.values())
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in vals):
            raise ValueError('Invalid action probability')
        new = {h: v+math.log(likelihoods[h]) if likelihoods[h] else -math.inf for h, v in self._logp.items()}
        self._commit(self.public, self.captured, new)
        self.uniform = self.uniform and len(set(vals)) == 1

    def hidden_capture(self):
        """Opponent captures an own dark piece; its identity is NOT an argument."""
        if not self.live:
            raise ValueError('No live dark piece')
        new = {}
        for h, v in self._logp.items():
            for t, (k, n) in enumerate(zip(self.public, h)):
                if k > n:
                    nxt = list(h); nxt[t] += 1; nxt = tuple(nxt)
                    term = v+math.log(k-n)-math.log(self.live)
                    new[nxt] = _logsum([new.get(nxt, -math.inf), term])
        self._commit(self.public, self.captured+1, new)

    def public_reveal(self, piece_type):
        if isinstance(piece_type, bool) or not isinstance(piece_type, Integral) or not 0 <= piece_type < len(self.public):
            raise ValueError('Invalid type')
        t = int(piece_type)
        if not self.live or not self.public[t]:
            raise ValueError('Impossible public reveal')
        public = list(self.public); public[t] -= 1
        new = {h: v+math.log(self.public[t]-h[t])-math.log(self.live)
               for h, v in self._logp.items() if self.public[t] > h[t]}
        self._commit(public, self.captured, new)

    def log_sequence_probability(self, revealed_counts):
        """Probability of any ONE ordered reveal sequence with these counts.

        Integrates sampling without replacement over h, with no multinomial
        factor; conditional ratios yield next-type probabilities.
        """
        draws = _counts(revealed_counts)
        if len(draws) != len(self.public) or sum(draws) > self.live:
            raise ValueError('Invalid reveal sequence')
        denominator = math.lgamma(self.live+1)-math.lgamma(self.live-sum(draws)+1)
        terms = []
        for h, v in self._logp.items():
            live = [k-n for k, n in zip(self.public, h)]
            if all(n <= c for n, c in zip(draws, live)):
                terms.append(v-denominator+sum(math.lgamma(c+1)-math.lgamma(c-n+1) for n, c in zip(draws, live)))
        return _logsum(terms)

    def predictive(self):
        if not self.live:
            return (0.,)*len(self.public)
        return tuple(math.exp(self.log_sequence_probability(tuple(int(i == t) for i in range(len(self.public)))))
                     for t in range(len(self.public)))

    def sample(self, rng):
        """Gumbel-max sampling preserves log support even after numerical underflow."""
        return max(self._logp, key=lambda h: self._logp[h]-math.log(-math.log(max(rng.random(), 1e-300))))

    def entropies(self):
        ledger = -math.fsum(math.exp(v)*v for v in self._logp.values())
        assignment = ledger+math.fsum(math.exp(v)*(math.lgamma(self.live+1)-sum(math.lgamma(k-n+1) for k, n in zip(self.public, h)))
                                      for h, v in self._logp.items())
        return {'private_ledger_nats': ledger, 'live_assignment_nats': assignment}
