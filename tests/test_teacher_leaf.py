import copy
from pathlib import Path
import re
import subprocess
import torch
import pytest
from research.policy_value import AZ, PhasePolicyValueNet
from research.teacher_leaf import TeacherStaticValue, TeacherLeafPUCT, observer_fen
from adapters.yuan_observer import reconstruct

ROOT = Path(__file__).resolve().parents[1]
NET = ROOT/'external/devinokeefe-jieqi/engine/nets/it014.jqn'
LIB = ROOT/'build/nnue-observer/libobserver_nnue.so'
pytestmark = pytest.mark.skipif(not LIB.exists(), reason='Build pinned observer NNUE library first')


def views():
    result = []
    for tokens in ([], ['b2b9R'], ['b2b9R', 'a6a5p']):
        for observer in (AZ.jq.RED, AZ.jq.BLACK):
            observed = [t+'n' if t == 'b2b9R' and observer == AZ.jq.RED else t for t in tokens]
            _, _, info = reconstruct(observed, viewer=observer)
            # Static evaluator side must be the observer whose ledger is used.
            info.side = observer
            result.append(info)
    full = copy.deepcopy(result[0])
    full.board = [0]*90
    full.board[4] = AZ.jq.make_piece(AZ.jq.TYPE_KING, AZ.jq.BLACK, 0)
    full.board[85] = AZ.jq.make_piece(AZ.jq.TYPE_KING, AZ.jq.RED, 0)
    full.board[40] = AZ.jq.make_piece(AZ.jq.TYPE_ROOK, AZ.jq.RED, 0)
    full.pool_known = [[0]*8, [0]*8]; full.pool_total_known = [0, 0]; full.unknown_removals = [0, 0]
    result.append(full)
    return result


def test_observer_bridge_matches_unchanged_binary_quantized_evaluation():
    teacher = TeacherStaticValue(NET); infos = views()
    commands = ['uci', 'setoption name EvalFile value '+str(NET), 'isready']
    for info in infos: commands += ['position fen '+observer_fen(info), 'eval']
    commands += ['quit']
    output = subprocess.run([str(ROOT/'external/devinokeefe-jieqi/engine/build/jqe')],
                            input='\n'.join(commands)+'\n', text=True, capture_output=True, check=True, timeout=30).stdout
    assert 'bad fen' not in output and 'load failed' not in output
    values = [int(x) for x in re.findall(r'^classical -?\d+ eval (-?\d+)$', output, re.M)]
    assert len(values) == len(infos)
    assert values == [teacher.centipawns(info) for info in infos]


def test_teacher_leaf_search_preserves_legal_visits_and_private_isolation():
    torch.set_num_threads(1); torch.manual_seed(7)
    net = PhasePolicyValueNet(4, 1).eval(); outputs = []
    teacher = TeacherStaticValue(NET)
    for poison in (False, True):
        state, belief, _ = reconstruct(['b2b9R'])
        if poison:
            belief.pool_counts[1-state.side] = [[0]*8, [0]*8]
            belief.pool_total[1-state.side] = [0, 0]
            belief.unknown_removals[1-state.side] = [99, 99]
        engine = TeacherLeafPUCT(net, seed=53, leaf_batch=2, teacher=teacher)
        policy, move, root = engine.search(state, belief, [], simulations=16, temperature=1e-6, return_root=True)
        assert move in state.legal_moves() and root.visits == 16
        assert sum(policy) == pytest.approx(1.)
        outputs.append((policy, move, root.value_sum, [(m, c.visits, c.value_sum) for m, c in root.children.items()]))
    assert outputs[0] == outputs[1]
