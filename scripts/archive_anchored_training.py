"""Archive a completed anchored update; no inference about playing strength."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import sha,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    folder=ROOT/args.run;out=ROOT/args.out
    if out.exists():raise ValueError('Refuse to overwrite training archive')
    manifest=json.loads((folder/'manifest.json').read_text());best=json.loads((folder/'best.json').read_text())
    if manifest['status']!='completed' or best['npz_sha256']!=sha(folder/'best.npz'):
        raise ValueError('Incomplete or changed selected model')
    log=json.loads((folder/'training-log.json').read_text())
    if len(log)!=manifest['epochs_completed'] or log[-1]['updates']!=manifest['updates']:
        raise ValueError('Incomplete training curve')
    if min(log,key=lambda x:x['validation']['loss'])['epoch']!=best['epoch']:
        raise ValueError('Checkpoint not selected by original unanchored validation loss')
    report={'schema':1,'status':'completed','scope':manifest['scope'],
        'plan_sha256':manifest['plan_sha256'],'initial_model_sha256':manifest['initial_network_sha256'],
        'observer_masks_sha256':manifest['observer_masks_sha256'],'policy_anchor_lambda':manifest['policy_anchor_lambda'],
        'training_games':len(manifest['training']),'training_positions':sum(x['positions'] for x in manifest['training']),
        'selection_games':len(manifest['selection_validation']),
        'selection_positions':sum(x['positions'] for x in manifest['selection_validation']),
        'epochs':manifest['epochs_completed'],'updates':manifest['updates'],'selected_checkpoint':best,
        'initial_validation':json.loads((folder/'initial-validation.json').read_text()),'training_curve':log,
        'limitations':['Frozen warm-policy legal KL added to search-visit/terminal-return fitting; not full adaptive Kickstarting.',
            'One training seed; same-data/selection comparison only, not independent generalization or a strength result.',
            'Actual paired matches against unanchored/warm/teacher controls remain required before any promotion.']}
    save_json(out,report)


if __name__=='__main__':main()
