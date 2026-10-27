"""Descriptive paired TRAIN-root proxy and entropy changes; no strength curves."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/20261027-policy-target-pruning-diagnostic.json'
OUT = ROOT / 'results/figures/policy_target_pruning_20261027'


def main():
    if any(OUT.with_suffix(s).exists() for s in ['.png', '.pdf', '.json']):
        raise ValueError('Refuse to overwrite diagnostic figures')
    report = json.loads(SOURCE.read_text())
    if report['status'] != 'completed' or len(report['cases']) != 32 or report['quarantined']:
        raise ValueError('Require the complete prespecified32-root diagnostic')
    phases = ['0-0', '1-4', '5-14', '15-30']
    labels = ['0', '1–4', '5–14', '15–30']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    colors = ['#246c83', '#b45c29']
    for panel, (axis, metric, title, ylabel) in enumerate(zip(axes,
            ['proxy', 'entropy'], ['Teacher-search proxy change', 'Target entropy reduction'],
            ['Pruned − raw (bounded proxy)', 'Raw − pruned (nats)'])):
        means = []
        for i, phase in enumerate(phases):
            rows = [r for r in report['cases'] if r['phase'] == phase]
            assert len(rows) == 8
            values = np.asarray([r['pruned_minus_raw_teacher_proxy'] if metric == 'proxy'
                else r['entropy']['raw'] - r['entropy']['pruned'] for r in rows])
            axis.scatter(i + np.linspace(-.13, .13, len(values)), values,
                         color=colors[panel], s=26, alpha=.8)
            means.append(values.mean())
        axis.scatter(range(4), means, marker='D', s=45, color='#222222', label='Stratum mean', zorder=4)
        axis.axhline(0, color='#999999', lw=.8)
        axis.set_xticks(range(4), labels)
        axis.set_xlabel('Physical live dark pieces (both colors)')
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Policy-target pruning: 32 fixed TRAIN roots, 8 per physical phase', fontsize=11)
    fig.savefig(OUT.with_suffix('.png'), dpi=180)
    fig.savefig(OUT.with_suffix('.pdf'), metadata={'Title': 'Policy-target pruning TRAIN-root diagnostic',
                                                 'Author': 'ChineseDarkChessAI research'})
    plt.close(fig)
    meta = {'schema': 1, 'source': str(SOURCE.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Descriptive individual-root changes and fixed-stratum means. Teacher search proxy is not truth or playing strength. Same32TRAIN roots, not independent generalization; no phase-interaction or monotonic-observability inference.',
        'artifacts': {str(OUT.with_suffix(s).relative_to(ROOT)): hashlib.sha256(OUT.with_suffix(s).read_bytes()).hexdigest()
                      for s in ['.png', '.pdf']}}
    OUT.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
