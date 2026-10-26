import copy,sys,tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research import policy_value as m
from yuan_observer import reconstruct

def test_record_encoder_matches_original_observable_prefix():
    for tokens in [[],['b2b9R'],['b2b9Rr','a9b9r']]:
        _,_,info=reconstruct(tokens)
        raw=m.records_from_info(info)
        actual=m.encode_records(torch.from_numpy(raw),canonical=False).numpy()[0]
        expected=m.AZ.encode_state_numpy(info,[])
        np.testing.assert_allclose(actual[:46],expected[:46],atol=1e-7)
        np.testing.assert_array_equal(actual[46:],np.zeros_like(actual[46:]))

def full_public_info():
    _,_,info=reconstruct([])
    info.board=[0]*90;info.board[4]=m.AZ.jq.make_piece(1,0,0);info.board[85]=m.AZ.jq.make_piece(1,1,0)
    info.board[49]=m.AZ.jq.make_piece(7,1,0)
    info.pool_known[0]=[0]*8;info.pool_known[1][7]=4
    info.pool_total_known=[0,0];info.unknown_removals=[0,14]
    return info

def test_full_public_invariance_does_not_switch_while_faces_remain():
    info=full_public_info();other=copy.deepcopy(info);other.pool_known[1]=[0]*8;other.unknown_removals=[0,0]
    raw=torch.from_numpy(np.concatenate([m.records_from_info(info),m.records_from_info(other)]))
    x=m.encode_records(raw)
    torch.testing.assert_close(x[0],x[1],rtol=0,atol=0)
    assert not torch.equal(m.encode_records(raw,canonical=False)[0],m.encode_records(raw,canonical=False)[1])
    _,_,partial=reconstruct([])
    assert m.encode_records(torch.from_numpy(m.records_from_info(partial)))[:,14:28].sum()>0

def test_phase_heads_preserve_initial_function_and_receive_only_their_phase_gradient():
    torch.set_num_threads(1)
    torch.manual_seed(9);base=m.PhasePolicyValueNet(8,1,False)
    torch.manual_seed(9);expert=m.PhasePolicyValueNet(8,1,True)
    _,_,partial=reconstruct([])
    raw=torch.from_numpy(np.concatenate([m.records_from_info(partial),m.records_from_info(full_public_info())]))
    x=m.encode_records(raw)
    base.eval();expert.eval()
    for a,b in zip(base(x),expert(x)):torch.testing.assert_close(a,b,rtol=0,atol=0)
    expert.zero_grad();expert(x)[1][1].backward()
    assert expert.value_fc1.weight.grad.abs().sum()==0
    assert expert.public_fc1.weight.grad.abs().sum()>0

def test_action_map_and_safe_array_interchange():
    assert len(np.unique(m.ACTION_NATIVE))==2550
    np.testing.assert_array_equal(m.ACTION_LOOKUP[m.ACTION_NATIVE],np.arange(2550))
    np.testing.assert_array_equal(m.MIRROR_ACTION[m.MIRROR_ACTION],np.arange(2550))
    torch.manual_seed(10);base=m.PhasePolicyValueNet(8,1)
    with tempfile.TemporaryDirectory() as directory:
        file=Path(directory)/'net.npz';m.export_npz(base,file)
        net=m.PhasePolicyValueNet(8,1,True);m.load_npz(net,file,common_initial=True)
        for key,value in base.state_dict().items():torch.testing.assert_close(value,net.state_dict()[key],rtol=0,atol=0)
        torch.testing.assert_close(net.value_fc1.weight,net.public_fc1.weight,rtol=0,atol=0)
