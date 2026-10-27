"""Protect full-budget and checkpoint-selection boundaries before match launch."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('spatial_queue', ROOT / 'scripts/queue_spatial_evaluation.py')
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def sample():
    report = json.loads((ROOT / 'results/20261027-spatial-policy-integration-dense2550.json').read_text())
    protocol = {'body': {'channels': 8, 'residual_blocks': 1},
                'training': {'epochs': 1, 'updates_each': 11}}
    return report, protocol


def test_completed_real_integration_and_full_budget_gate():
    report, protocol = sample()
    queue.validate_training(report, protocol, 'dense2550')
    full = copy.deepcopy(protocol)
    full['training'].update(epochs=12, updates_each=56652)
    with pytest.raises(ValueError, match='budget'):
        queue.validate_training(report, full, 'dense2550')
    report['status'] = 'running'
    with pytest.raises(ValueError, match='incomplete'):
        queue.validate_training(report, protocol, 'dense2550')


def test_selection_cannot_use_last_epoch_or_nonfinite_loss():
    report, protocol = sample()
    protocol['training'].update(epochs=2, updates_each=22)
    report.update(epochs=2, updates=22, selected_epoch=2)
    second = copy.deepcopy(report['training_curve'][0])
    second.update(epoch=2, updates=22)
    second['validation']['loss'] += 1
    report['training_curve'].append(second)
    with pytest.raises(ValueError, match='Checkpoint'):
        queue.validate_training(report, protocol, 'dense2550')
    report['selected_epoch'] = 1
    queue.validate_training(report, protocol, 'dense2550')
    second['validation']['loss'] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        queue.validate_training(report, protocol, 'dense2550')


def test_missing_epoch_and_wrong_decoder_fail_before_matches():
    report, protocol = sample()
    with pytest.raises(ValueError, match='profile'):
        queue.validate_training(report, protocol, 'relative50')
    report['training_curve'] = []
    with pytest.raises(ValueError, match='epoch curve'):
        queue.validate_training(report, protocol, 'dense2550')
