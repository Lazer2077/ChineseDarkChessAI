"""Archive a complete frozen diagnostic, including negative/quarantined roots."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.neural_replay import sha, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    path, out = ROOT / args.plan, ROOT / args.out
    if out.exists():
        raise ValueError('Refuse to overwrite a pruning diagnostic archive')
    plan = json.loads(path.read_text())
    for name, expected in plan['sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen pruning input changed: ' + name)
    source = ROOT / plan['out'] / 'report.json'
    report = json.loads(source.read_text())
    if report['status'] != 'completed' or report['plan_sha256'] != sha(path):
        raise ValueError('Incomplete or unfrozen pruning diagnostic')
    if not report['all_planned_cases_accounted']:
        raise ValueError('Missing planned roots')
    expected = {(r['game'], r['ply']) for r in plan['cases']}
    actual = [(r['game'], r['ply']) for r in report['cases'] + report['quarantined']]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Repeated or substituted roots')
    report['source_report_sha256'] = sha(source)
    save_json(out, report)


if __name__ == '__main__':
    main()
