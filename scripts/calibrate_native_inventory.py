"""Old opponent-only games, public-input search likelihoods, separate hidden labels."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'external/devinokeefe-jieqi'))
from research.native_opponent_policy import FixedDepthOpponent, hypothetical_fen, score_policy
from research.private_inventory import PrivateInventory
from jieqi.native import NativeRules, NativeGame, GameStatus
from jieqi.setup import Color, Kind, Piece, Setup, square

INITIAL = 'XXXXKXXXX'+'.'*10+'X'+'.'*5+'X'+'.'+'X.X.X.X.X'+'.'*18+'x.x.x.x.x'+'.'+'x'+'.'*5+'x'+'.'*10+'xxxxkxxxx'
TYPES = 'RNBACP'


def save(path, data):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(data, indent=2)+'\n'); tmp.replace(path)


def public_events(game):
    # A deliberate projection: private captured-type fields and setup seed do
    # not cross into public reconstruction or opponent likelihood calculation.
    return [{'move': row['m'], 'reveal': row['r']} for row in game['moves']]


def public_frames(events, rules):
    public = [[2, 2, 2, 2, 2, 5] for _ in range(2)]
    with NativeGame(rules, INITIAL, Color.RED) as game:
        for ply, event in enumerate(events):
            snap = game.snapshot(); board = snap.board; actor = 0 if snap.turn == Color.RED else 1
            if snap.status != GameStatus.ONGOING: raise ValueError('Public continuation after terminal')
            frm, to = square(event['move'][:2]), square(event['move'][2:])
            moved, victim = board[frm], board[to]
            legal = rules.legal_moves(board, snap.turn)
            if event['move'] not in legal: raise ValueError('Public illegal move')
            reveal = event['reveal']
            if (moved in 'Xx') != (reveal is not None): raise ValueError('Public reveal mismatch')
            yield {'ply': ply, 'board': board, 'actor': actor, 'public': [r.copy() for r in public],
                   'legal': legal, 'move': event['move'], 'reveal': reveal,
                   'dark_capture': victim in 'Xx', 'no_capture': snap.no_capture_plies}
            game.advance(event['move'], Kind(reveal) if reveal else None)
            if reveal: public[actor][TYPES.index(reveal)] -= 1
        if game.snapshot().status == GameStatus.ONGOING: raise ValueError('Missing terminal public outcome')


def scoring_labels(game):
    """Referee-only truth; return h before each move, never used by policy code."""
    truth = list(Setup.from_seed(int(game['seed'])).board)
    captured = [[0]*6 for _ in range(2)]; labels = []
    for ply, row in enumerate(game['moves']):
        labels.append([h.copy() for h in captured])
        frm, to = square(row['m'][:2]), square(row['m'][2:])
        moved, victim = truth[frm], truth[to]
        actor = Color.RED if ply%2 == 0 else Color.BLACK
        if moved is None or moved.color != actor: raise ValueError('Seeded mover mismatch')
        if row['r'] != (moved.kind.value if moved.hidden else None): raise ValueError('Seeded reveal mismatch')
        if row['c'] != (victim.kind.value if victim else None): raise ValueError('Seeded capture mismatch')
        if victim and victim.hidden:
            captured[0 if victim.color == Color.RED else 1][TYPES.index(victim.kind.value)] += 1
        truth[to] = Piece(moved.color, moved.kind, False); truth[frm] = None
    return labels


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True); args = parser.parse_args()
    path = Path(args.plan); plan = json.loads(path.read_text()); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    def verify():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest: raise ValueError('Plan changed')
        for relative, h in plan['sha256'].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != h: raise ValueError('Pinned input changed: '+relative)
    verify(); out = ROOT/plan['out']; out.mkdir(parents=True, exist_ok=False)
    games = [json.loads(line) for line in gzip.decompress((ROOT/plan['raw_games']).read_bytes()).splitlines()]
    games.sort(key=lambda g: (g['pair'], g['red']))
    selected = games[:plan['games']]
    rules = NativeRules(ROOT/'external/devinokeefe-jieqi/build/release/lib/libjieqi_rules.so')
    report = {'schema': 1, 'status': 'running', 'scope': plan['scope'], 'plan_sha256': digest, 'games': [], 'samples': []}
    save(out/'report.json', report)
    with FixedDepthOpponent(ROOT/plan['binary'], ROOT/plan['network'], depth=plan['depth']) as model:
        for game_index, game in enumerate(selected):
            labels = scoring_labels(game)
            teacher = 0 if game['red'] == 'candidate' else 1; observer = 1-teacher
            posterior, reference = PrivateInventory((2, 2, 2, 2, 2, 5)), PrivateInventory((2, 2, 2, 2, 2, 5))
            used = 0; queries = 0; positions = 0; larger_support_skips = 0
            for frame in public_frames(public_events(game), rules):
                positions += 1
                if frame['actor'] == teacher and len(posterior.log_probabilities) > 1 and used < plan['decisions_per_game']:
                    if len(posterior.log_probabilities) > plan['max_support']:
                        larger_support_skips += 1
                    else:
                        predictions = {}; likelihoods = {}
                        for h in posterior.log_probabilities:
                            fen = hypothetical_fen(frame['board'], teacher, frame['public'], observer, h, frame['no_capture'])
                            scores = model.scores(fen, frame['legal']); queries += len(scores)
                            p = score_policy(scores, plan['temperature'], plan['epsilon'])
                            likelihoods[h] = p[frame['move']]
                            predictions[','.join(map(str, h))] = {'action_scores_cp': scores, 'observed_action_probability': likelihoods[h]}
                        posterior.condition_action(likelihoods)
                        # Label is consulted only AFTER all predictions/update.
                        label = tuple(labels[frame['ply']][observer])
                        gain = posterior.log_probabilities[label]-reference.log_probabilities[label]
                        report['samples'].append({'game_index': game_index, 'pair': game['pair'], 'red': game['red'],
                            'ply': frame['ply'], 'teacher_color': teacher, 'support': len(likelihoods), 'label_h': label,
                            'observed_move': frame['move'], 'predictions': predictions, 'log_score_gain_nats': gain})
                        used += 1
                if frame['actor'] == observer and frame['reveal']:
                    t = TYPES.index(frame['reveal']); posterior.public_reveal(t); reference.public_reveal(t)
                if frame['actor'] == teacher and frame['dark_capture']:
                    posterior.hidden_capture(); reference.hidden_capture()
            report['games'].append({'pair': game['pair'], 'red': game['red'], 'positions_public_replayed': positions,
                                    'scored_decisions': used, 'forced_searches': queries, 'large_support_skips': larger_support_skips,
                                    'seeded_private_labels_verified': True})
            verify(); save(out/'report.json', report)
            print(json.dumps(report['games'][-1]), flush=True)
    gains = [s['log_score_gain_nats'] for s in report['samples']]
    report.update(status='completed', summary={'scored_decisions': len(gains),
        'mean_log_score_gain_nats': math.fsum(gains)/len(gains) if gains else None,
        'warning': 'Correlated sparse reused baseline trajectories. Fixed-depth/history-reset soft search policy approximates a timed/history-aware engine. Not independent calibration or playing strength.'})
    verify(); save(out/'report.json', report)


if __name__ == '__main__': main()
