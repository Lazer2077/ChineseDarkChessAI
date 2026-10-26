import copy
import io
import json
import os
from pathlib import Path
import pytest
import torch
from research.fixed_work_profile import Settings, FixedInput, constructor, public_board
from research.fixed_work_contract import INITIAL_BOARD, public_key, validate_record
from research.policy_value import PhasePolicyValueNet, AZ
from research.neural_search import ObservationPUCT
from research.neural_replay import load_episode, fresh_game
from adapters.yuan_observer import reconstruct

ROOT = Path(__file__).resolve().parents[1]


def root_summary(result):
    policy, move, root = result
    return policy, move, root.value_sum, [(m, c.visits, c.value_sum) for m, c in root.children.items()]


def test_original_go_is_witnessed_while_work_is_explicitly_forced(tmp_path):
    settings = Settings('puct', 8, tmp_path)
    lines = list(FixedInput(io.StringIO('uci\ngo wtime 1000 btime 900\nquit\n'), settings))
    assert lines == ['uci\n', 'go nodes 8\n', 'quit\n']
    assert settings.pending_go == 'wtime 1000 btime 900'


def test_partial_profile_exactly_preserves_original_search_and_public_board(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(7)
    net = PhasePolicyValueNet(4, 1).eval(); state, belief, _ = reconstruct([])
    assert public_board(state) == INITIAL_BOARD
    original = ObservationPUCT(net, seed=13, leaf_batch=8)
    expected = original.search(state, belief, [], simulations=16, temperature=0., return_root=True)
    settings = Settings('public-halving', 16, tmp_path); settings.pending_go = 'movetime 1'
    engine = constructor(settings)(net, seed=13, leaf_batch=8)
    actual = engine.search(state, belief, [], simulations=16, temperature=0., return_root=True)
    settings.close()
    assert root_summary(actual) == root_summary(expected)
    row = json.loads((tmp_path/f'{os.getpid()}.jsonl').read_text())
    assert validate_record(row, 16, 'public-halving') == 'puct'
    assert row['public_key'] == public_key(INITIAL_BOARD, 'r', [])
    wrong = copy.deepcopy(row); wrong['fully_public'] = True
    with pytest.raises(ValueError, match='phase gate'): validate_record(wrong, 16, 'public-halving')
    wrong = copy.deepcopy(row); wrong['completed_simulations'] = 17
    with pytest.raises(ValueError, match='simulation budget'): validate_record(wrong, 16, 'public-halving')


def test_full_public_profile_switches_and_records_actual_visits(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(11)
    data, meta = load_episode(ROOT/'runs/neural-selfplay-pilot-01/episodes/game-000001.npz')
    state, belief, rng = fresh_game(meta['seed']); history = []
    for raw in data['moves'][:65]:
        move = tuple(map(int, raw)); AZ.im.apply_move_and_update_belief(state, belief, move, rng=rng); history.append(move)
    assert 'X' not in public_board(state) and 'x' not in public_board(state)
    settings = Settings('public-halving', 16, tmp_path); settings.pending_go = 'nodes 16'
    engine = constructor(settings)(PhasePolicyValueNet(4, 1).eval(), seed=17, leaf_batch=8)
    engine.search(state, belief, history, simulations=16, temperature=0., return_root=True)
    settings.close()
    row = json.loads((tmp_path/f'{os.getpid()}.jsonl').read_text())
    assert validate_record(row, 16, 'public-halving') == 'halving'
    assert row['ply'] == 65 and sum(e['visits'] for e in row['root_edges'].values()) == 16
    wrong = copy.deepcopy(row); wrong['root_edges'][wrong['selected_move']]['visits'] += 1
    with pytest.raises(ValueError): validate_record(wrong, 16, 'public-halving')
