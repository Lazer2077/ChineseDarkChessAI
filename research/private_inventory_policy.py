"""Counterfactual opponent action probabilities from public facts plus h.

Count-only policy approximation: capture order and full belief distributions are
not encoded. This is not an exact model of an arbitrary opponent strategy.
"""
from dataclasses import fields
import numpy as np
import torch
from research.policy_value import AZ, records_from_info, encode_records


def counterfactual_records(root_info, root_color, inventory):
    if root_info.side == root_color:
        raise ValueError('Only opponent decisions carry opponent-policy evidence')
    public = root_info.public_pool_known
    if public is None or tuple(public[root_color][t] for t in AZ.jq.POOL_TYPES) != inventory.public:
        raise ValueError('Require matching public inventory')
    states = list(inventory.log_probabilities)
    records = []
    for h in states:
        data = {f.name: getattr(root_info, f.name) for f in fields(AZ.im.InformationState)}
        # Replace BOTH inventories, including the actor's own public pool. Our
        # privately known captures of enemy pieces must never reach that actor.
        data['pool_known'] = [row.copy() for row in public]
        for t, n in zip(AZ.jq.POOL_TYPES, h):
            data['pool_known'][root_color][t] -= n
        data['unknown_removals'] = [sum(row)-total for row, total in zip(data['pool_known'], root_info.pool_total_known)]
        if min(data['unknown_removals']) < 0:
            raise ValueError('Public live counts disagree')
        records.append(records_from_info(AZ.im.InformationState(**data)))
    return states, np.concatenate(records)


def action_likelihoods(net, root_info, root_color, inventory, legal_moves, observed_move, temperature=2., epsilon=.1):
    if not temperature > 0 or not 0 < epsilon <= 1:
        raise ValueError('Require positive temperature and robust uniform mixture')
    ids = [AZ.move_to_index(m) for m in legal_moves]
    action = AZ.move_to_index(observed_move)
    if not ids or len(set(ids)) != len(ids) or action not in ids:
        raise ValueError('Invalid public legal support or observed move')
    states, raw = counterfactual_records(root_info, root_color, inventory)
    device = next(net.parameters()).device
    net.eval()
    with torch.inference_mode():
        logits, _ = net(encode_records(torch.from_numpy(raw).to(device)))
        policy = torch.softmax(logits[:, ids].double()/temperature, dim=1)
        likelihood = ((1-epsilon)*policy[:, ids.index(action)]+epsilon/len(ids)).cpu().tolist()
    return dict(zip(states, likelihood))
