import math
import numpy as np
import pytest
import torch
from research.expert_value_diagnostic import search_score_proxy,observer_input_keys
from research.policy_value import encode_records


def test_score_proxy_keeps_perspective_and_rejects_ambiguous_mate():
    assert search_score_proxy({'kind':'cp','value':860,'depth':3})==math.tanh(1)
    assert search_score_proxy({'kind':'cp','value':-860,'depth':3})==-math.tanh(1)
    assert search_score_proxy({'kind':'mate','value':-2,'depth':4})==-1
    for score in [{'kind':'mate','value':0,'depth':1},{'kind':'cp','value':True,'depth':1},{'kind':'cp','value':12,'depth':0}]:
        with pytest.raises(ValueError):search_score_proxy(score)


def test_key_matches_public_canonical_encoder_and_ignores_targets():
    raw=np.zeros((2,112),dtype=np.uint8);raw[1,91]=2;raw[1,103:111]=255
    assert observer_input_keys(raw)[0]==observer_input_keys(raw)[1]
    x=encode_records(torch.from_numpy(raw));assert torch.equal(x[0],x[1])
    raw[1,111]=1;assert observer_input_keys(raw)[0]!=observer_input_keys(raw)[1]


def test_dark_inventory_and_actor_remain_distinct_inputs():
    raw=np.zeros((2,112),dtype=np.uint8)
    # a0 is a valid dark home. Keep enough unknown types to cover it.
    raw[:,0]=8;raw[:,91]=1;raw[1,91]=2
    assert observer_input_keys(raw)[0]!=observer_input_keys(raw)[1]
    x=encode_records(torch.from_numpy(raw));assert not torch.equal(x[0],x[1])
    raw[1]=raw[0];raw[1,90]=1;assert observer_input_keys(raw)[0]!=observer_input_keys(raw)[1]
