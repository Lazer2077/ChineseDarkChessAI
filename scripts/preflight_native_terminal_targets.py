"""Check matched inputs/visits and independently projected native return signs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from research.native_terminal_targets import load_split, validate_split, annotation
from research.neural_replay import sha, save_json


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-plan', required=True)
    p.add_argument('--native-plan', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise ValueError('Refuse to overwrite target data preflight')
    source, native = [json.loads((ROOT / path).read_text()) for path in [args.source_plan, args.native_plan]]
    generation = json.loads((ROOT / source['generation_plan']).read_text())
    folder = ROOT / generation['out']
    manifest = json.loads((folder / 'manifest.json').read_text())
    for plan in [source, native]:
        for path, expected in plan['sha256'].items():
            if sha(ROOT / path) != expected:
                raise ValueError('Frozen target data input changed: ' + path)
        validate_split(plan, generation, plan['training_games'], plan['selection_validation_games'])
    annotations = annotation(native, ROOT)
    rows = []
    for role in ['training_games', 'selection_validation_games']:
        ids = source[role]
        if ids != native[role]:
            raise ValueError('Prefix profiles use different games')
        raw, pi, z, metadata = load_split(source, folder, ids, manifest)
        nraw, npi, nz, nmetadata = load_split(native, folder, ids, manifest)
        if not np.array_equal(raw, nraw) or not np.array_equal(pi, npi):
            raise ValueError('Prefix observations or visit policies differ between labels')
        expected = np.concatenate([np.asarray([
            annotations[i]['native_red_return'] * (1 if ply % 2 == 0 else -1)
            for ply in range(annotations[i]['native_terminal_ply'])], dtype=np.float32) for i in ids])
        if not np.array_equal(nz, expected) or not np.isfinite(pi).all() or not np.all(pi.sum(1) == 1):
            raise ValueError('Native return perspective or normalized visits differ')
        if len(raw) != sum(annotations[i]['native_terminal_ply'] for i in ids):
            raise ValueError('Postterminal source roots admitted')
        rows.append({'role': role, 'games': len(ids), 'positions': len(raw),
                     'observations_sha256': digest(raw), 'normalized_visits_sha256': digest(pi),
                     'source_returns_sha256': digest(z), 'native_returns_sha256': digest(nz),
                     'changed_return_positions': int(np.count_nonzero(z != nz)),
                     'bitwise_matched_observations_and_visits': True,
                     'native_returns_match_independent_turn_parity': True})
    save_json(out, {'schema': 1, 'status': 'completed', 'roles': rows,
                   'source_plan_sha256': sha(ROOT / args.source_plan),
                   'native_plan_sha256': sha(ROOT / args.native_plan),
                   'script_sha256': sha(__file__),
                   'scope': 'Real-data target/control boundary checks on identical native-terminal prefixes. No fit, model selection, new strength games or promotion.'})
    print(json.dumps({'status': 'completed', 'roles': rows}), flush=True)


if __name__ == '__main__':
    main()
