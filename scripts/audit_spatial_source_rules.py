"""Training-only public native replay of completed source-rule self-play."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.neural_replay import load_episode, save_json, sha
from research.policy_value import AZ
# Both upstreams name a module "jieqi". Load the referee's unchanged package
# under its own package namespace; never replace Yuan's live module binding.
PACKAGE = '_cdca_public_rule_audit_referee'
DIRECTORY = ROOT / 'external/devinokeefe-jieqi/jieqi'
SPEC = importlib.util.spec_from_file_location(PACKAGE, DIRECTORY / '__init__.py',
                                             submodule_search_locations=[str(DIRECTORY)])
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[PACKAGE] = MODULE
SPEC.loader.exec_module(MODULE)
NATIVE = importlib.import_module(PACKAGE + '.native')
SETUP = importlib.import_module(PACKAGE + '.setup')
GameStatus, NativeGame, NativeRules = NATIVE.GameStatus, NATIVE.NativeGame, NATIVE.NativeRules
Color, Kind, start_squares = SETUP.Color, SETUP.Kind, SETUP.start_squares


LETTERS = dict(zip(AZ.jq.POOL_TYPES, 'RNBACP'))
LETTERS[AZ.jq.TYPE_KING] = 'K'
LETTERS[8] = 'X'
ACTION_NAMES = [AZ.jq.format_move_uci(move) for move in AZ.LEGAL_MOVE_TABLE]


def board_text(record):
    board = []
    for code in record[:90]:
        code = int(code)
        kind = code & 15
        if not kind:
            board.append('.')
        else:
            letter = LETTERS[kind]
            board.append(letter.lower() if code & 16 else letter)
    return ''.join(board)


def public_start():
    board = ['.'] * 90
    for color, letter in [(Color.RED, 'X'), (Color.BLACK, 'x')]:
        for square in start_squares(color):
            board[square] = letter
    board[4], board[85] = 'K', 'k'
    return ''.join(board)


def audit_game(rules, data, metadata, limit):
    source_n = metadata['positions']
    row = {'source_positions': source_n,
           'source_red_return': AZ.outcome_value_for_player(metadata['outcome'], AZ.jq.RED),
           'source_reason': metadata['reason'], 'native_replayed_moves': 0,
           'native_illegal_visit_roots': 0, 'native_illegal_visits': 0,
           'counter_mismatch_roots': 0, 'first_illegal_visit_root': None,
           'first_counter_mismatch_root': None, 'illegal_chosen_move': None}
    with NativeGame(rules, public_start(), Color.RED, no_capture_limit=limit) as game:
        for ply, raw_move in enumerate(data['moves']):
            snap = game.snapshot()
            if snap.status is not GameStatus.ONGOING:
                break
            record = data['records'][ply]
            expected_turn = Color.RED if int(record[90]) == 0 else Color.BLACK
            if snap.board != board_text(record) or snap.turn is not expected_turn:
                raise ValueError('Unexpected source/native public-board or turn serialization difference')
            if snap.no_capture_plies != int(record[111]):
                row['counter_mismatch_roots'] += 1
                if row['first_counter_mismatch_root'] is None:
                    row['first_counter_mismatch_root'] = ply
            legal = set(rules.legal_moves(snap.board, snap.turn))
            visits = data['visit_counts'][ply]
            invalid = [i for i in visits.nonzero()[0] if ACTION_NAMES[i] not in legal]
            if invalid:
                row['native_illegal_visit_roots'] += 1
                row['native_illegal_visits'] += sum(int(visits[i]) for i in invalid)
                if row['first_illegal_visit_root'] is None:
                    row['first_illegal_visit_root'] = {'ply': ply,
                        'actions': [ACTION_NAMES[i] for i in invalid]}
            move = AZ.jq.format_move_uci(tuple(map(int, raw_move)))
            if move not in legal:
                row['illegal_chosen_move'] = {'ply': ply, 'move': move}
                break
            reveal = int(data['reveals'][ply, 0])
            game.advance(move, Kind(LETTERS[reveal]) if reveal else None)
            row['native_replayed_moves'] += 1
        terminal = game.snapshot()
        native_value = {GameStatus.DRAW: 0, GameStatus.RED_LOSES: -1,
                        GameStatus.BLACK_LOSES: 1}.get(terminal.status)
    row.update(native_status=terminal.status.name, native_red_return=native_value,
               native_terminal_ply=terminal.plies if native_value is not None else None,
               source_positions_after_native_terminal=source_n - terminal.plies if native_value is not None else 0,
               terminal_value_matches_source=(native_value == row['source_red_return']) if native_value is not None else None,
               same_terminal_ply=(terminal.plies == source_n) if native_value is not None else False)
    row['complete_native_equivalent_trace'] = (row['illegal_chosen_move'] is None and
        row['native_illegal_visit_roots'] == 0 and row['counter_mismatch_roots'] == 0 and
        row['same_terminal_ply'] and row['terminal_value_matches_source'])
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    path = ROOT / args.plan
    plan = json.loads(path.read_text())
    digest = sha(path)
    out = ROOT / plan['out']
    if out.exists():
        raise ValueError('Refuse to overwrite public-rule diagnostic')

    def verify():
        if sha(path) != digest:
            raise ValueError('Public-rule audit plan changed')
        for name, expected in plan['sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Frozen rule-audit input changed: ' + name)

    verify()
    rules = NativeRules(ROOT / plan['library'])
    rows = []
    for index in plan['training_games']:
        file = ROOT / plan['episodes'] / f'game-{index:06d}.npz'
        data, metadata = load_episode(file)
        rows.append({'index': index, 'source_npz_sha256': sha(file),
                     **audit_game(rules, data, metadata, plan['native_no_capture_limit'])})
    verify()
    report = {'schema': 1, 'status': 'completed', 'plan_sha256': digest,
        'scope': plan['scope'], 'games': len(rows),
        'source_positions': sum(r['source_positions'] for r in rows),
        'native_replayed_moves': sum(r['native_replayed_moves'] for r in rows),
        'native_equivalent_complete_traces': sum(r['complete_native_equivalent_trace'] for r in rows),
        'native_ended_earlier_games': sum(r['source_positions_after_native_terminal'] > 0 for r in rows),
        'source_positions_after_native_terminal': sum(r['source_positions_after_native_terminal'] for r in rows),
        'native_ongoing_at_source_end_games': sum(r['native_status'] == 'ONGOING' and r['illegal_chosen_move'] is None for r in rows),
        'native_terminal_value_disagreements': sum(r['terminal_value_matches_source'] is False for r in rows),
        'illegal_chosen_move_games': sum(r['illegal_chosen_move'] is not None for r in rows),
        'native_illegal_visit_roots': sum(r['native_illegal_visit_roots'] for r in rows),
        'counter_mismatch_roots': sum(r['counter_mismatch_roots'] for r in rows),
        'per_game': rows, 'limitations': plan['limitations']}
    save_json(out, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['per_game', 'limitations']}), flush=True)


if __name__ == '__main__':
    main()
