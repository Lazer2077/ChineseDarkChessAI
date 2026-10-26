import copy,sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.policy_value import AZ,PhasePolicyValueNet
from research.neural_search import ObservationPUCT
from research.neural_replay import episode,audit_episode,save_episode,load_episode

@pytest.fixture(scope='module')
def complete_episode():
    torch.set_num_threads(1);torch.manual_seed(73)
    net=PhasePolicyValueNet(8,1);net.eval()
    return episode(ObservationPUCT(net,leaf_batch=2),8250001,4)

def test_real_episode_and_safe_roundtrip_replay_private_observations(complete_episode,tmp_path):
    data,metadata=complete_episode;audit=audit_episode(data,metadata)
    assert audit['positions_replayed']==len(data['records'])
    assert audit['private_dark_captures']>0 and audit['fully_public_positions']>0
    save_episode(tmp_path,0,data,metadata)
    loaded,meta=load_episode(tmp_path/'game-000000.npz')
    assert all(np.array_equal(data[key],loaded[key]) for key in data)
    assert meta['audit']==audit

def test_private_inventory_and_terminal_label_corruption_are_rejected(complete_episode):
    original,metadata=complete_episode
    for key,index in [('records',(0,91)),('values',(0,))]:
        data={k:v.copy() for k,v in original.items()};data[key][index]+=1
        with pytest.raises(ValueError):audit_episode(data,metadata)

def test_illegal_visit_mass_and_incomplete_root_are_rejected(complete_episode):
    original,metadata=complete_episode
    data={k:v.copy() for k,v in original.items()}
    from research.neural_replay import fresh_game
    state,_,_=fresh_game(metadata['seed']);legal={AZ.move_to_index(m) for m in state.legal_moves()}
    illegal=next(i for i in range(AZ.MOVE_SIZE) if i not in legal)
    data['visit_counts'][0]=0;data['visit_counts'][0,illegal]=metadata['simulations']
    with pytest.raises(ValueError):audit_episode(data,metadata)
    data={k:v.copy() for k,v in original.items()};data['root_visits'][0]-=1
    with pytest.raises(ValueError):audit_episode(data,metadata)

def test_resource_truncation_has_no_terminal_target():
    torch.set_num_threads(1);torch.manual_seed(73)
    net=PhasePolicyValueNet(8,1);net.eval()
    with pytest.raises(ValueError,match='Resource-truncated'):
        episode(ObservationPUCT(net,leaf_batch=2),8250001,4,max_plies=2)
