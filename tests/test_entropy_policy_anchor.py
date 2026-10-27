import math
import json
import hashlib
import numpy as np
import pytest
import torch
from research.policy_anchor import legal_teacher_kl
from research.entropy_policy_anchor import weighted_legal_teacher_kl
from research.entropy_policy_anchor import load_entropy_weights
from research.privileged_training import fresh_fixed_game,training_pair


def test_unit_weights_bitwise_match_constant_loss_and_gradients():
    student=torch.tensor([[0.,1.,2.],[2.,1.,0.]],requires_grad=True)
    teacher=torch.tensor([[1.,2.,0.],[0.,2.,1.]],requires_grad=True)
    mask=torch.tensor([[True,True,False],[True,False,True]])
    reference=legal_teacher_kl(student,teacher,mask)
    grad=torch.autograd.grad(reference,student)[0]
    candidate=weighted_legal_teacher_kl(student,teacher,mask,torch.ones(2))
    assert torch.equal(reference,candidate)
    assert torch.equal(grad,torch.autograd.grad(candidate,student)[0])


def test_zero_weight_has_no_anchor_gradient_without_batch_renormalization():
    student=torch.zeros((2,3),requires_grad=True)
    teacher=torch.tensor([[math.log(3),0.,1e9],[math.log(3),0.,-1e9]],requires_grad=True)
    mask=torch.tensor([[True,True,False],[True,True,False]])
    loss=weighted_legal_teacher_kl(student,teacher,mask,torch.tensor([2.,0.]))
    expected=.75*math.log(1.5)+.25*math.log(.5)
    assert loss.item()==pytest.approx(expected,abs=1e-7)
    loss.backward();assert teacher.grad is None
    assert student.grad[0].tolist()==pytest.approx([-.25,.25,0.],abs=1e-7)
    assert not student.grad[1].any()
    with pytest.raises(ValueError,match='coefficients'):
        weighted_legal_teacher_kl(student,teacher,mask,torch.tensor([-1.,2.]))


def test_weight_loader_recomputes_observer_features_and_rejects_wrong_row_order(tmp_path,monkeypatch):
    import research.entropy_policy_anchor as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    state,belief,_=fresh_fixed_game(1003);initial,_=training_pair(state,belief)
    certain=initial.copy();certain[:90]=0;certain[0]=8;certain[91:103]=0;certain[91]=1
    raw=np.stack([initial,certain]);meta=[{'index':0,'positions':2,'sha256':'fixture'}]
    folder=tmp_path/'u';folder.mkdir();report=folder/'report.json';arrays=folder/'diagnostic.npz'
    def write(weights):
        np.savez_compressed(arrays,candidate_mean_matched_weight=np.asarray(weights,dtype=np.float64))
        report.write_text(json.dumps({'status':'completed','episodes':meta,
            'arrays_sha256':hashlib.sha256(arrays.read_bytes()).hexdigest()}))
    plan={'entropy_weight_mode':'mean-matched-observer-entropy','uncertainty_data':'u'}
    write([2.,0.]);assert load_entropy_weights(plan,meta,raw,'cpu').tolist()==[2.,0.]
    raw[:,103:111]=255
    assert load_entropy_weights(plan,meta,raw,'cpu').tolist()==[2.,0.]
    write([0.,2.])
    with pytest.raises(ValueError,match='observable-only'):load_entropy_weights(plan,meta,raw,'cpu')
