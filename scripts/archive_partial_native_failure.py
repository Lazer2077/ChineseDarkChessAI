"""Archive an incomplete screen and its void without repeating its failed slot."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.neural_replay import sha, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--partial-audit', required=True)
    p.add_argument('--match-report', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    folder, out = ROOT / args.run, ROOT / args.out
    if out.exists() or (ROOT / args.match_report).exists():
        raise ValueError('Refuse to overwrite incomplete-match evidence')
    audit = json.loads((ROOT / args.partial_audit).read_text())
    summary = json.loads((folder / 'cdca-summary.json').read_text())
    void = (folder / 'void.jsonl').read_bytes()
    if (not audit['partial_match'] or not audit['all_planned_attempt_slots_accounted'] or
            audit['games'] != 63 or audit['complete_pairs'] != 31 or audit['void_attempts'] != 1 or
            hashlib.sha256(void).hexdigest() != audit['void_raw_sha256'] or
            summary['valid_technical_run'] or summary['games'] != 63 or summary['void_attempts'] != 1):
        raise ValueError('Partial-result or failure boundary differs')
    digest = hashlib.sha256(void).hexdigest()
    file = ROOT / 'results/raw' / (folder.name + '-void-' + digest[:12] + '.jsonl.gz')
    if file.exists():
        raise ValueError('Refuse to overwrite void evidence')
    file.write_bytes(gzip.compress(void, mtime=0))
    subprocess.run([sys.executable, str(ROOT / 'scripts/archive_native_results.py'),
        '--out', args.match_report, '--raw-dir', 'results/raw', args.run], check=True, cwd=ROOT)
    save_json(out, {'schema':1,'status':'archived-incomplete','match_report':args.match_report,
        'match_report_sha256':sha(ROOT / args.match_report),'partial_audit':args.partial_audit,
        'partial_audit_sha256':sha(ROOT / args.partial_audit),
        'void_archive':{'path':str(file.relative_to(ROOT)),'uncompressed_sha256':digest},
        'observed_failures':['One zero-ply candidate process exit; upstream discards engine stderr.',
            'Harness worker raises OSError28 while flushing the void record; planned retry is not reached.'],
        'scope':'63complete games/31complete pairs plus one retained startup void,not a valid64-game screen. Do not refill or rerun the missing slot as clean confirmation. The two remaining original protocol contrasts continue unchanged in a separate controller.',
        'causality_limit':'Storage exhaustion is observed in the harness. The unrecorded engine exception cannot be attributed conclusively to that exhaustion.',
        'promotion_supported':False})


if __name__ == '__main__':
    main()
