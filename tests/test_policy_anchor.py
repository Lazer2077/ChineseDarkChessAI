import math
import numpy as np
import pytest
import torch
from research.policy_value import AZ
from research.policy_anchor import legal_teacher_kl,observer_legal_mask
from research.privileged_training import fresh_fixed_game


def test_masked_kl_exact_value_gradients_and_teacher_detachment():
    student=torch.tensor([[0.,0.,1e9]],requires_grad=True)
    teacher=torch.tensor([[math.log(3),0.,-1e9]],requires_grad=True)
    mask=torch.tensor([[True,True,False]])
    loss=legal_teacher_kl(student,teacher,mask)
    assert loss.item()==pytest.approx(.75*math.log(1.5)+.25*math.log(.5),abs=1e-7)
    loss.backward()
    assert teacher.grad is None
    assert student.grad.tolist()[0]==pytest.approx([-.25,.25,0.],abs=1e-7)
    assert legal_teacher_kl(teacher,teacher,mask).item()==0
    with pytest.raises(ValueError,match='nonempty'):legal_teacher_kl(student,teacher,torch.zeros_like(mask))


def test_observer_mask_after_private_capture_and_visit_mismatch_rejects():
    state,belief,rng=fresh_fixed_game(999)
    move=tuple(AZ.jq.name_to_square(x) for x in ('b2','b9'))
    AZ.im.apply_move_and_update_belief(state,belief,move,rng=rng)
    info=belief.info_state(state,state.side)
    mask=observer_legal_mask(info)
    assert set(np.flatnonzero(mask))=={AZ.move_to_index(m) for m in state.legal_moves()}
    counts=np.zeros(AZ.MOVE_SIZE,dtype=np.uint32);counts[np.flatnonzero(mask)[0]]=16
    assert np.array_equal(observer_legal_mask(info,counts),mask)
    counts[np.flatnonzero(~mask)[0]]=1
    with pytest.raises(ValueError,match='Stored visit support'):observer_legal_mask(info,counts)
