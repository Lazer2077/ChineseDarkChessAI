"""Fixed completed-simulation actor profiles and public-only budget witnesses."""
import json
import math
import os
from pathlib import Path
from research.neural_search import ObservationPUCT
from research.sequential_halving import HalvingPUCT
from research.policy_value import AZ
from research.fixed_work_contract import public_key


def public_board(state):
    def letter(pc):
        if not pc: return '.'
        value = 'X' if AZ.jq.is_dark(pc) else 'KRNBACP'[AZ.jq.piece_type(pc)-1]
        return value if AZ.jq.piece_color(pc) == AZ.jq.RED else value.lower()
    return ''.join(letter(pc) for r in range(9, -1, -1) for pc in state.board[r*9:r*9+9])


class Settings:
    def __init__(self, strategy, simulations, witness_dir):
        if strategy not in ('puct', 'halving', 'public-halving') or type(simulations) is not int or not 1 <= simulations <= 65536:
            raise ValueError('Invalid fixed-work profile')
        self.strategy, self.simulations, self.witness_dir = strategy, simulations, Path(witness_dir)
        self.pending_go = None; self.calls = 0; self.fd = None

    def write(self, record):
        if self.fd is None:
            self.witness_dir.mkdir(parents=True, exist_ok=True)
            self.fd = os.open(self.witness_dir/f'{os.getpid()}.jsonl', os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
        data = (json.dumps(record, separators=(',', ':'))+'\n').encode(); offset = 0
        while offset < len(data): offset += os.write(self.fd, data[offset:])

    def close(self):
        if self.fd is not None: os.close(self.fd); self.fd = None


class FixedInput:
    def __init__(self, stream, settings): self.stream, self.settings = stream, settings
    def __iter__(self):
        for line in self.stream:
            if line.split()[:1] == ['go']:
                self.settings.pending_go = line.strip()[2:].strip()
                yield 'go nodes '+str(self.settings.simulations)+'\n'
            else: yield line


def constructor(settings):
    class FixedWorkPUCT(HalvingPUCT):
        def search(self, root_state, belief, history, **kwargs):
            if kwargs.get('simulations') != settings.simulations or self.deadline is not None or settings.pending_go is None:
                raise ValueError('Require explicit forced work and witnessed original go request')
            requested_root = kwargs.get('return_root', False); kwargs['return_root'] = True
            board = public_board(root_state); fully_public = not ('X' in board or 'x' in board)
            use_halving = settings.strategy == 'halving' or (settings.strategy == 'public-halving' and fully_public)
            before = self.leaf_evals_total
            if use_halving:
                result = HalvingPUCT.search(self, root_state, belief, history, gumbel_seed=0, noise_scale=0., candidates=16, **kwargs)
            else:
                result = ObservationPUCT.search(self, root_state, belief, history, **kwargs)
            policy, move, root = result
            if root.visits != settings.simulations or sum(c.visits for c in root.children.values()) != root.visits:
                raise ValueError('Incomplete actual completed simulations')
            if move not in root.children or root.children[move].visits <= 0 or abs(sum(policy)-1.) > 1e-7:
                raise ValueError('Invalid returned move or visit policy')
            if any(not math.isfinite(c.value_sum) or abs(c.value_sum) > c.visits+1e-6 for c in root.children.values()):
                raise ValueError('Outstanding virtual value credits or nonfinite backup')
            settings.calls += 1
            actor = 'r' if root_state.side == AZ.jq.RED else 'b'
            moves = [AZ.jq.format_move_uci(m) for m in history]
            settings.write({'schema': 1, 'pid': os.getpid(), 'ordinal': settings.calls, 'ply': len(history),
                'strategy': settings.strategy, 'actual_method': 'halving' if use_halving else 'puct',
                'requested_simulations': settings.simulations, 'completed_simulations': root.visits,
                'outer_go': settings.pending_go, 'public_board': board, 'actor': actor,
                'public_key': public_key(board, actor, moves), 'fully_public': fully_public,
                'selected_move': AZ.jq.format_move_uci(move), 'visit_mass': sum(policy),
                'nn_leaf_evals_excluding_root': self.leaf_evals_total-before,
                'root_edges': {AZ.jq.format_move_uci(m): {'visits': c.visits, 'next_player_value_sum': c.value_sum}
                               for m, c in root.children.items()}})
            return result if requested_root else result[:2]
    return FixedWorkPUCT
