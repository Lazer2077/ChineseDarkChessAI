import hashlib
import json
import numpy as np
import pytest
import torch
from research.policy_value import AZ,CONTRACT,MIRROR_ACTION,encode_records,export_npz
from research.spatial_policy_value import DISPLACEMENTS,DISPLACEMENT_INDEX,POLICY_GATHER,POLICY_MAPPING_SHA256,gather_policy,make_model,load_model


def test_all_actions_unique_and_gather_gradients_exclude_unused_geometry():
    grid=torch.arange(4500,dtype=torch.float32).reshape(1,50,10,9).requires_grad_()
    logits=gather_policy(grid)
    assert logits.shape==(1,2550) and len(set(POLICY_GATHER))==2550
    for i,(src,dst) in enumerate(AZ.LEGAL_MOVE_TABLE):
        plane=DISPLACEMENT_INDEX[(dst//9-src//9,dst%9-src%9)]
        assert logits[0,i]==grid[0,plane,src//9,src%9]
    logits.sum().backward();assert int(grid.grad.count_nonzero())==2550
    assert bool((grid.grad.flatten()[torch.as_tensor(POLICY_GATHER)]==1).all())


def test_codec_commutes_with_geometric_reflections_without_asserting_rule_symmetry():
    grid=torch.randn(2,50,10,9)
    mirror=[DISPLACEMENT_INDEX[(dr,-dc)] for dr,dc in DISPLACEMENTS]
    rotated=[DISPLACEMENT_INDEX[(-dr,-dc)] for dr,dc in DISPLACEMENTS]
    logits=gather_policy(grid)
    assert torch.equal(gather_policy(grid[:,mirror].flip(-1)),logits[:,MIRROR_ACTION])
    rotation=[AZ.move_to_index((89-f,89-t)) for f,t in AZ.LEGAL_MOVE_TABLE]
    assert torch.equal(gather_policy(grid[:,rotated].flip((-2,-1))),logits[:,rotation])


def test_common_initialization_value_function_and_safe_roundtrip(tmp_path):
    torch.manual_seed(20261029);dense=make_model('dense2550',8,1).eval()
    torch.manual_seed(20261029);relative=make_model('relative50',8,1).eval()
    old=dense.state_dict();new=relative.state_dict()
    assert all(torch.equal(old[k],new[k]) for k in old.keys()&new.keys())
    raw=torch.zeros(2,112,dtype=torch.uint8);raw[1,90]=1;x=encode_records(raw)
    with torch.inference_mode():
        pd,vd=dense(x);pr,vr=relative(x)
    assert pd.shape==pr.shape==(2,2550) and torch.equal(vd,vr)
    file=tmp_path/'best.npz';export_npz(relative,file)
    meta={'policy_head':'relative50','policy_mapping_sha256':POLICY_MAPPING_SHA256,'input_contract':CONTRACT,
          'channels':8,'blocks':1,'phase_experts':False,'npz_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    p=tmp_path/'best.json';p.write_text(json.dumps(meta));loaded,_=load_model(tmp_path)
    with torch.inference_mode():pp,vp=loaded(x)
    assert torch.equal(pp,pr) and torch.equal(vp,vr)
    meta['policy_mapping_sha256']='wrong';p.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='geometry'):load_model(tmp_path)
