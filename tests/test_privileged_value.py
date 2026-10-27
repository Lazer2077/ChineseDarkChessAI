import numpy as np
import pytest
import torch
import sys
from pathlib import Path
from research.policy_value import AZ, PhasePolicyValueNet, encode_records
from research.privileged_training import fresh_fixed_game, training_pair
from research.privileged_value import PrivilegedValueNet, encode_value_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_privileged_value import metrics, summarize_test


def test_truth_planes_align_with_observer_and_mirror_and_zero_control():
    state, belief, _ = fresh_fixed_game(996)
    raw, truth = training_pair(state, belief)
    r = torch.from_numpy(raw.copy()[None]); t = torch.from_numpy(truth.copy()[None])
    x = encode_value_inputs(r, t, True); zero = encode_value_inputs(r, t, False)
    assert x.shape == (1, 250, 10, 9)
    assert torch.equal(x[:, :238], encode_records(r))
    assert not zero[:, 238:].any()
    assert x[:, 238:].sum() == 30
    for sq, pc in enumerate(state.board):
        if not pc or not AZ.jq.is_dark(pc): continue
        slot = list(AZ.jq.POOL_TYPES).index(int(truth[sq]))
        assert x[0, 238+AZ.jq.piece_color(pc)*6+slot, sq//9, sq%9] == 1
        assert x.flip(-1)[0, 238+AZ.jq.piece_color(pc)*6+slot, sq//9, 8-sq%9] == 1
    bad = t.clone(); bad[0, int(np.flatnonzero(truth)[0])] = 0
    with pytest.raises(ValueError, match='geometry'): encode_value_inputs(r, bad, True)


def test_same_capacity_initial_function_and_privileged_gradient():
    torch.set_num_threads(1); torch.manual_seed(997)
    state, belief, _ = fresh_fixed_game(997); raw, truth = training_pair(state, belief)
    r = torch.from_numpy(raw.copy()[None]); t = torch.from_numpy(truth.copy()[None])
    warm = PhasePolicyValueNet(4, 1).eval()
    symmetric = PrivilegedValueNet(warm).eval(); privileged = PrivilegedValueNet(warm).eval()
    x = encode_value_inputs(r,t,True); z = encode_value_inputs(r,t,False)
    with torch.inference_mode():
        _, expected = warm(encode_records(r))
        assert torch.equal(symmetric(z), expected)
        assert torch.equal(privileged(x), expected)
    assert sum(p.numel() for p in symmetric.parameters()) == sum(p.numel() for p in privileged.parameters())
    for a,b in zip(symmetric.parameters(),privileged.parameters()): assert torch.equal(a,b)
    symmetric(z).sum().backward(); privileged(x).sum().backward()
    assert not symmetric.network.conv.weight.grad[:,238:].any()
    assert privileged.network.conv.weight.grad[:,238:].abs().sum() > 0
    assert all(not p.requires_grad for n,p in privileged.network.named_parameters() if n.startswith('policy_'))


def test_game_cluster_weights_and_no_small_sample_positive_claim():
    groups=np.asarray([0,0,0,1]); z=np.zeros(4)
    predictions={'symmetric':np.asarray([1.,1.,1.,0.]),'privileged':np.zeros(4)}
    m=metrics(predictions['symmetric'],z,groups)
    assert m['position_weighted_mse']==.75 and m['game_weighted_mse']==.5
    result=summarize_test(predictions,z,groups,np.ones((4,90),dtype=np.uint8),998,1000)
    assert result['game_weighted_mse_reduction_symmetric_minus_privileged']==.5
    assert not result['privilege_predictive_benefit_supported']
    assert result['phases'][0]['positions']==0
