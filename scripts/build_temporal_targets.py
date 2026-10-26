"""Relabel complete observer-specific trajectories for a controlled temporal-target ablation.

Synthetic score fields encode learning targets, never actual engine evaluations.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def segment_games(records, expected_games):
    ply = records['ply'].astype(np.int64)
    if not len(ply):
        raise ValueError('Empty trajectories')
    boundary = np.diff(ply) != 1
    if {'board', 'move'}.issubset(records.dtype.names):
        prev, after = records['board'][:-1], records['board'][1:]
        moves = records['move'][:-1].astype(np.int64)
        src, dst = moves & 127, (moves >> 7) & 127
        valid = (src < 90) & (dst < 90) & (src != dst)
        src, dst = np.minimum(src, 89), np.minimum(dst, 89)
        row = np.arange(len(prev))
        changed = prev != after
        outside = changed.sum(1) - changed[row, src] - changed[row, dst]
        piece, revealed = prev[row, src], after[row, dst]
        normal = revealed == piece
        hidden = ((piece & 15) == 8) & ((revealed >> 4) == (piece >> 4)) & ((revealed & 15) >= 2) & ((revealed & 15) <= 7)
        follows = valid & (outside == 0) & (after[row, src] == 0) & (piece != 0) & (normal | hidden)
        boundary |= ~follows
    starts = np.r_[0, np.flatnonzero(boundary) + 1]
    # Generator writes one contiguous, consecutive-ply buffer per completed game.
    # A valid transition changes only source/destination and publicly reveals a
    # hidden mover. There can be no within-game break. Matching the game count proves
    # no game boundary was accidentally merged and no empty game was omitted.
    if len(starts) != expected_games:
        raise ValueError(f'Cannot certify game boundaries: {len(starts)} != {expected_games}')
    ends = np.r_[starts[1:], len(ply)]
    if np.any(records['stm'] != ply % 2):
        raise ValueError('Acting side does not match the recorded ply')
    for start, end in zip(starts, ends):
        red_result = np.where(records['stm'][start:end] == 0, records['result'][start:end], -records['result'][start:end])
        if not np.isin(red_result, [-1, 0, 1]).all() or np.any(red_result != red_result[0]):
            raise ValueError('Inconsistent terminal labels within game')
    return starts, ends


def temporal_targets(records, expected_games, tau=.5, scale=430):
    if not 0 <= tau <= 1:
        raise ValueError('tau must be in [0,1]')
    starts, ends = segment_games(records, expected_games)
    prob = 1 / (1 + np.exp(-records['score'].astype(np.float64) / scale))
    outcome = (records['result'].astype(np.float64) + 1) / 2
    bootstrap = outcome.copy()
    has_next = np.zeros(len(records), dtype=bool)
    for start, end in zip(starts, ends):
        if end - start > 2:
            # Two plies later is the SAME observer, including that observer's
            # updated private capture information. Never negate opponent values.
            bootstrap[start:end - 2] = prob[start + 2:end]
            has_next[start:end - 2] = True
    return .7 * prob + .3 * ((1 - tau) * outcome + tau * bootstrap), has_next


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--expected-games', required=True, type=int)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / 'external/devinokeefe-jieqi/engine/train/train.py'
    spec = importlib.util.spec_from_file_location('target_backbone', source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    data = Path(a.data)
    if data.suffix != '.bin' or data.stat().st_size % 112:
        raise SystemExit('Only a complete .bin input is accepted')
    marker = data.with_suffix('.complete.json')
    attestation = json.loads(marker.read_text())
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    if digest != attestation['sha256'] or attestation['recipe']['games'] != a.expected_games or attestation['rejected_games'] != 0:
        raise SystemExit('Require matching complete generation with zero rejected games')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    records = np.fromfile(data, dtype=m.REC)
    raw = records.view(np.uint8).reshape(-1, 112)
    keep = m.keep_mask(raw)
    meta = {'schema': 1, 'source_sha256': digest, 'games': a.expected_games,
            'raw_records': len(records), 'retained_records': int(keep.sum()),
            'scale': 430, 'control_lambda': .7,
            'interpretation': 'Synthetic score encodes target probability; use trainer --lam=1.0. Same-observer two-ply bootstrap, not opponent-observation negamax. Validation selection must use the control target for BOTH variants.', 'variants': {}}
    for label, tau in [('control', 0.), ('temporal', .5)]:
        target, has_next = temporal_targets(records, a.expected_games, tau=tau)
        clipped = np.clip(target, 1e-6, 1 - 1e-6)
        encoded = np.rint(430 * (np.log(clipped) - np.log1p(-clipped))).astype(np.int16)
        relabeled = records[keep].copy()
        relabeled['score'] = encoded[keep]
        dest = out / (label + '.bin')
        relabeled.tofile(dest)
        restored = 1 / (1 + np.exp(-encoded[keep].astype(np.float64) / 430))
        error = float(np.max(np.abs(restored - target[keep])))
        if error > .0003:
            raise RuntimeError('Excessive target quantization error')
        meta['variants'][label] = {'tau': tau, 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
                                   'same_observer_bootstrap_rows': int((has_next & keep).sum()),
                                   'maximum_target_encoding_error': error}
    (out / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
