"""Archive a completed conditional-prediction diagnostic, never a chess victory."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.neural_replay import sha,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    run=ROOT/args.run;report=json.loads((run/'report.json').read_text())
    if report['status']!='completed':raise ValueError('Require complete diagnostic')
    predictions=run/'test-predictions.npz'
    if sha(predictions)!=report['test_predictions_sha256']:raise ValueError('Test predictions changed')
    target=ROOT/args.out
    if target.exists():raise ValueError('Refuse to overwrite archived diagnostic')
    artifact=ROOT/'results/raw'/('privileged-value-predictions-'+sha(predictions)[:12]+'.npz')
    if artifact.exists():raise ValueError('Prediction archive already exists')
    shutil.copyfile(predictions,artifact)
    report.pop('pid',None)
    report['test_predictions_archive']=str(artifact.relative_to(ROOT))
    for name in ('symmetric','privileged'):
        report['profiles'][name]['training_curve']=json.loads((run/name/'training-log.json').read_text())
    report['limitations']=['Conditional terminal-return prediction, not an actor update or playing-strength gain.',
        'History-omitting observer features plus live-dark identity labels are not a full history-state critic.',
        'Only one training seed; test uncertainty is clustered by whole game, not positions.',
        'Both critics are training-only artifacts with incompatible input contracts for ordinary playing actors.']
    save_json(target,report)


if __name__=='__main__':main()
