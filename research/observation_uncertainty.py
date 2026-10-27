"""Live-type assignment entropy under the existing exchangeable count prior.

This is exact for the declared finite-inventory model, not an exact adversarial
posterior after conditioning on an opponent's action-selection strategy.
"""
from functools import lru_cache
import math
import numpy as np
from research.private_inventory import PrivateInventory

INITIAL_COUNTS=(2,2,2,2,2,5)
INITIAL_TWO_SIDE_ENTROPY=2*(math.lgamma(16)-sum(math.lgamma(k+1) for k in INITIAL_COUNTS))


@lru_cache(maxsize=20000)
def live_assignment_entropy(counts,unknown_removed):
    inventory=PrivateInventory(counts,unknown_removed)
    return max(0.,inventory.entropies()['live_assignment_nats'])


def record_uncertainty(raw):
    """Only public board and actor-known inventories are read from byte records.

    Entropy marginalizes unknown own losses; no actual true-type assignment,
    opponent-private ledger, rollout outcome, future reveal or seed is accepted.
    A type-certain face-down position remains distinct from publicly revealed.
    """
    raw=np.asarray(raw)
    if raw.dtype!=np.uint8 or raw.ndim!=2 or raw.shape[1]!=112:
        raise ValueError('Require observer records as(N,112)uint8')
    board=raw[:,:90];dark=(board&15)==8;colors=board>>4
    if (colors>1).any():raise ValueError('Invalid public colour')
    counts=raw[:,91:103].view(np.int8).reshape(-1,2,6)
    if (counts<0).any() or (counts>np.asarray(INITIAL_COUNTS)).any():
        raise ValueError('Invalid actor-known inventories')
    live=np.stack([(dark&(colors==c)).sum(1) for c in (0,1)],axis=1)
    missing=counts.sum(2)-live
    if (missing<0).any():raise ValueError('Known inventory cannot cover live dark pieces')
    entropy=np.asarray([sum(live_assignment_entropy(tuple(map(int,counts[i,c])),int(missing[i,c]))
        for c in (0,1)) for i in range(len(raw))],dtype=np.float64)
    if (entropy>INITIAL_TWO_SIDE_ENTROPY+1e-9).any():raise ValueError('Impossible entropy bound')
    return {'entropy_nats':entropy,'normalized_entropy':np.clip(entropy/INITIAL_TWO_SIDE_ENTROPY,0.,1.),
            'live_dark_count':live.sum(1),'fully_public':live.sum(1)==0,'type_certain':entropy<1e-12}
