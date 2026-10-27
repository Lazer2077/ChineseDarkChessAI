"""Training-only fixed-deal labels; never import this module from playing actors.

Preserve face-down move geometry. True identities are separate labels, not
substituted into the public board. A fixed deal is sampled before play, without
conditioning it on future moves, revelations, or outcomes.
"""
import random
import numpy as np
from research.policy_value import AZ, records_from_info

CONTRACT = 'cdca-training-only-fixed-deal-dark-labels-v1'


def fresh_fixed_game(seed):
    deal = AZ.jq.materialize_dark_types(random.Random('cdca-privileged-deal:'+str(seed)))
    rng = random.Random(seed)
    state = AZ.jq.GameState.initial(rng=rng, dark_types=deal)
    return state, AZ.im.init_belief_for_game(state), rng


def dark_truth(state):
    """Yuan square order; zero on empty/public squares, true type on live darks."""
    if state.dark_types is None:
        raise ValueError('Require an ex ante materialized deal, not a lazy/future-completed trajectory')
    truth = np.zeros(90, dtype=np.uint8)
    for sq, pc in enumerate(state.board):
        if pc and AZ.jq.is_dark(pc):
            t = state.dark_types.get(sq)
            if t not in AZ.jq.POOL_TYPES:
                raise ValueError('Missing live-dark identity')
            truth[sq] = t
    for c in (AZ.jq.RED, AZ.jq.BLACK):
        for t in AZ.jq.POOL_TYPES:
            count = sum(int(truth[sq]) == t for sq, pc in enumerate(state.board)
                        if pc and AZ.jq.is_dark(pc) and AZ.jq.piece_color(pc) == c)
            if count != state.pool[c][t]:
                raise ValueError('Fixed deal does not match remaining environment inventory')
    truth.setflags(write=False)
    return truth


def actor_projection(state):
    """Remove the actual assignment before handing a state to existing PUCT.

Existing search still builds particles from actor information. This additional
boundary prevents even incidental access to the referee's assignment map.
It does not constitute a proof that every possible rules/history path is blind
to all other hidden environment quantities.
"""
    projected = state.clone()
    projected.dark_types = None
    return projected


def training_pair(state, belief):
    info = belief.info_state(state, state.side)
    return records_from_info(info)[0], dark_truth(state)
