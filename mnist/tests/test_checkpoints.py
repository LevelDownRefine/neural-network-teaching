import pytest
import torch
from utils import checkpoints


def test_existing_experiment_is_not_overwritten(tmp_path):
    (tmp_path / 'best.pt').write_bytes(b'original')
    with pytest.raises(FileExistsError, match='--resume or a new --tag'):
        checkpoints.prepare_output(tmp_path, resume=False)
    assert (tmp_path / 'best.pt').read_bytes() == b'original'
    checkpoints.prepare_output(tmp_path, resume=True)


def test_resume_recovers_best_after_interrupted_pair_save(tmp_path, monkeypatch):
    previous = {'epoch': 1, 'model': {'weight': torch.tensor([1.])}}
    current = {'epoch': 2, 'model': {'weight': torch.tensor([2.])}}
    checkpoints.atomic_save(previous, tmp_path / 'best.pt')
    real_save = checkpoints.atomic_save

    def interrupt_best(value, path):
        if path.name == 'best.pt':
            raise OSError('simulated interruption')
        real_save(value, path)

    monkeypatch.setattr(checkpoints, 'atomic_save', interrupt_best)
    with pytest.raises(OSError):
        checkpoints.save_training({'epoch': 2, 'best_checkpoint': current}, tmp_path)
    assert torch.load(tmp_path/'best.pt', weights_only=True)['epoch'] == 1
    monkeypatch.setattr(checkpoints, 'atomic_save', real_save)
    state = torch.load(tmp_path/'latest.pt', weights_only=True)
    checkpoints.restore_best(state, tmp_path)
    assert torch.load(tmp_path/'best.pt', weights_only=True)['epoch'] == 2


def test_legacy_checkpoint_can_resume(tmp_path):
    checkpoints.atomic_save({'epoch': 3}, tmp_path/'best.pt')
    assert checkpoints.restore_best({'epoch': 4}, tmp_path)['epoch'] == 3
