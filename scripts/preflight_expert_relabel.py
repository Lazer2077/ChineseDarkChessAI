"""Frozen, training-only expert-label feasibility on learner-generated histories."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import load_episode,fresh_game,sha,save_json
from research.policy_value import AZ
from research.teacher_leaf import observer_fen
from research.observer_expert import ObserverHistoryExpert,observer_token


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    path=ROOT/args.plan;plan=json.loads(path.read_text());digest=sha(path)
    def verify():
        if sha(path)!=digest:raise ValueError('Expert relabel plan changed')
        for name,h in plan['sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Frozen expert relabel input changed: '+name)
    verify();out=ROOT/plan['out'];out.mkdir(parents=True,exist_ok=False)
    report={'schema':1,'status':'running','scope':plan['scope'],'plan_sha256':digest,
            'nodes_per_query':plan['nodes'],'episodes':[]}
    save_json(out/'report.json',report)
    # No learner model or privileged assignment is loaded by this experiment.
    with ObserverHistoryExpert(ROOT/plan['teacher_binary'],ROOT/plan['teacher_net'],depth=1) as expert:
        for game in plan['games']:
            source=ROOT/plan['episodes']/f'game-{game:06d}.npz'
            data,metadata=load_episode(source)
            state,belief,rng=fresh_game(metadata['seed']);histories=[[],[]];labels=[]
            row={'game':game,'source_npz_sha256':sha(source),'source_positions':metadata['positions'],
                 'status':'running','roots_checked':0,'private_dark_captures':0,'teacher_action_disagreements':0}
            for ply,raw_move in enumerate(data['moves']):
                move=tuple(map(int,raw_move));actor=state.side
                info=belief.info_state(state,actor)
                # Full history replay is essential: never reset to the current FEN
                # to hide an incompatible history or a repetition/chase constraint.
                try:
                    result=expert.analyse(histories[actor],observer_fen(info),plan['nodes'])
                    native_legal=set(result['legal_moves'])
                    source_legal={AZ.jq.format_move_uci(m) for m in state.legal_moves()}
                    if native_legal!=source_legal:
                        row['legal_difference']={'native_only':sorted(native_legal-source_legal),
                                                 'source_only':sorted(source_legal-native_legal)}
                        raise ValueError('Native and source legal action sets differ')
                except ValueError as error:
                    row.update(status='quarantined',failure_ply=ply,failure=str(error))
                    # No prefix is admitted as training data after a failed episode.
                    break
                choice=result['bestmove'];score=result['last_completed_score']
                labels.append({'ply':ply,'actor':actor,'expert_move':choice,
                    'learner_move':AZ.jq.format_move_uci(move),'score':score})
                row['roots_checked']+=1
                row['teacher_action_disagreements']+=choice!=AZ.jq.format_move_uci(move)
                undo=AZ.im.apply_move_and_update_belief(state,belief,move,rng=rng)
                if tuple(map(int,data['reveals'][ply]))!=(undo.moved_reveal,undo.captured_reveal):
                    raise ValueError('Replayed source reveal differs')
                row['private_dark_captures']+=bool(undo.captured_reveal)
                for viewer in (AZ.jq.RED,AZ.jq.BLACK):
                    histories[viewer].append(observer_token(move,undo,viewer))
            else:
                outcome,reason=state.game_outcome()
                if outcome!=metadata['outcome'] or reason!=metadata['reason']:
                    raise ValueError('Replayed source terminal differs')
                # Terminal outcomes belong to the source rule profile. They are
                # not expert value labels or evidence of native-rule equivalence.
                row.update(status='compatible-roots',source_terminal_reason=reason,
                           native_terminal_equivalence_verified=False)
                label_path=out/f'game-{game:06d}-labels.json'
                save_json(label_path,{'source_npz_sha256':sha(source),'labels':labels})
                row['labels_sha256']=sha(label_path)
            report['episodes'].append(row);verify();save_json(out/'report.json',report)
            print(json.dumps(row),flush=True)
    report.update(status='completed',compatible_games=sum(x['status']=='compatible-roots' for x in report['episodes']),
        quarantined_games=sum(x['status']=='quarantined' for x in report['episodes']),
        checked_roots=sum(x['roots_checked'] for x in report['episodes']))
    verify();save_json(out/'report.json',report)


if __name__=='__main__':main()
