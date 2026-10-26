"""Uncertainty of future-relevant identities under the actor's urn prior.

Counts include unknown off-board removals. Piece types on labeled remaining
squares are sampled without replacement; this is not a policy-conditioned posterior.
"""
from functools import lru_cache
import math


@lru_cache(maxsize=None)
def _entropy(counts, squares):
    if not squares:
        return 0.
    total = sum(counts)
    result = 0.
    for i, count in enumerate(counts):
        if count:
            probability = count / total
            after = list(counts)
            after[i] -= 1
            result += probability * (-math.log2(probability) + _entropy(tuple(after), squares - 1))
    return max(0., result)


def identity_entropy(counts, squares):
    """Joint Shannon bits for the ordered on-board identities, not dead pieces."""
    values = tuple(counts)
    if any(int(x) != x for x in values):
        raise ValueError('Urn multiplicities must be integers')
    counts = tuple(int(x) for x in values)
    if any(x < 0 for x in counts) or int(squares) != squares or not 0 <= squares <= sum(counts):
        raise ValueError('Inconsistent observable inventory')
    return _entropy(counts, int(squares))


INITIAL_ENTROPY = 2 * identity_entropy((2, 2, 2, 2, 2, 5), 15)


def public_counts(records, starts, ends):
    """Recover public urn bounds from each actor's OWN-color upper bounds.

    The next actor's own-color bounds cannot change when the current actor
    moves: only the mover reveals, while hidden capture identities stay private.
    At a game's final row, undo the previous mover's public reveal instead.
    Singleton games cannot certify both public bags and are marked missing.
    """
    import numpy as np
    public = records['urn'].copy().astype(np.int64)
    available = np.zeros(len(records), dtype=bool)
    for start, end in zip(starts, ends):
        for i in range(start, end):
            actor = int(records['stm'][i])
            other = 1 - actor
            if i + 1 < end:
                public[i, other] = records['urn'][i + 1, other]
                available[i] = True
            elif i > start:
                public[i, other] = records['urn'][i - 1, other]
                move = int(records['move'][i - 1])
                src, dst = move & 127, (move >> 7) & 127
                if (int(records['board'][i - 1, src]) & 15) == 8:
                    revealed = int(records['board'][i, dst]) & 15
                    if not 2 <= revealed <= 7:
                        raise ValueError('Missing public reveal in trajectory')
                    public[i, other, revealed - 2] -= 1
                available[i] = True
    if np.any(public[available] < records['urn'][available]):
        raise ValueError('Public counts cannot reveal more than actor-private counts')
    return public, available
