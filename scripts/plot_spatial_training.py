"""Export supervised-learning curves from completed frozen decoder reports."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dense', required=True)
    p.add_argument('--relative', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    prefix = Path(args.out)
    paths = [prefix.with_suffix(suffix) for suffix in ['.png', '.pdf', '.json']]
    if any(path.exists() for path in paths):
        raise ValueError('Refuse to overwrite training curves')
    reports = [json.loads(Path(path).read_text()) for path in [args.dense, args.relative]]
    for report, head in zip(reports, ['dense2550', 'relative50']):
        if (report['status'] != 'completed' or report['policy_head'] != head or
                report['epochs'] != 12 or report['updates'] != 56652 or
                len(report['training_curve']) != 12):
            raise ValueError('Require both completed full-budget decoder reports')
    if any(reports[0][key] != reports[1][key] for key in ['training', 'selection_validation']):
        raise ValueError('Different replay or selection data')
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    colors = ['#1764a0', '#c45b12']
    labels = ['Dense decoder, 12.80M parameters', 'Relative decoder, 5.45M parameters']
    for report, color, label in zip(reports, colors, labels):
        curve = report['training_curve']
        updates = [row['updates'] / 1000 for row in curve]
        axes[0].plot(updates, [r['train_policy_ce'] for r in curve], ':', color=color, alpha=.7)
        axes[0].plot(updates, [r['validation']['policy_ce'] for r in curve], color=color, label=label)
        axes[1].plot(updates, [r['train_value_mse'] for r in curve], ':', color=color, alpha=.7)
        axes[1].plot(updates, [r['validation']['value_mse'] for r in curve], color=color)
        axes[2].plot(updates, [100 * r['validation']['teacher_action_top1'] for r in curve], color=color)
        selected = curve[report['selected_epoch'] - 1]
        for axis, value in zip(axes, [selected['validation']['policy_ce'],
                                      selected['validation']['value_mse'],
                                      100 * selected['validation']['teacher_action_top1']]):
            axis.scatter(selected['updates'] / 1000, value, marker='o', s=35,
                         facecolors='white', edgecolors=color, zorder=3)
    for axis, title, ylabel in zip(axes,
            ['Teacher-action prediction', 'Mixed search/outcome value', 'Unmasked teacher-action accuracy'],
            ['Cross entropy', 'Mean squared error', 'Top-1 accuracy (%)']):
        axis.set(title=title, xlabel='Optimizer updates (thousands)', ylabel=ylabel)
        axis.grid(alpha=.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(.5, 1.01))
    fig.text(.5, .005,
        'Solid: selection; dotted: training. Circles: selected by selection CE + MSE. '
        'Single training seed; supervised pretraining, no playing-strength conclusion.',
        ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .91))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=180, bbox_inches='tight')
    fig.savefig(paths[1], bbox_inches='tight')
    plt.close(fig)
    metadata = {'schema': 1, 'status': 'completed',
        'scope': 'Curves from two completed one-seed supervised decoder fits. Common TRAIN/selection data; initialization excluded from curves and checkpoint selection. Top1 uses all2550geometric logits without actual legal masking. No independent-game, RL or strength evidence.',
        'sources': {path: sha(path) for path in [args.dense, args.relative]},
        'script_sha256': sha(__file__),
        'selected': [{'head': r['policy_head'], 'epoch': r['selected_epoch'],
                       'model_sha256': r['model_sha256']} for r in reports],
        'figures': {path.as_posix(): sha(path) for path in paths[:2]}}
    paths[2].write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'status': 'completed', 'figures': list(metadata['figures'])}))


if __name__ == '__main__':
    main()
