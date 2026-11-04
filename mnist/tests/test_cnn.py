from pathlib import Path

import torch

from models.m5 import ModelM5
from models.sopcnn import ModelSOPCNN


def test_saved_cnn_checkpoint_loads_and_predicts():
    path = Path(__file__).resolve().parents[1] / 'results' / 'cnn_m5_kair' / 'best.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = ModelM5().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    with torch.no_grad():
        output = model(torch.zeros(2, 1, 28, 28))
    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()
    assert torch.allclose(output.exp().sum(dim=1), torch.ones(2), atol=1e-6)


def test_sopcnn_outputs_valid_class_probabilities():
    model = ModelSOPCNN().eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 1, 28, 28))
    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()
    assert torch.allclose(output.exp().sum(dim=1), torch.ones(2), atol=1e-6)


def test_small_m5_matches_tssl_parameter_budget():
    model = ModelM5((1, 14, 30, 37, 48, 69)).eval()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 210376
    with torch.no_grad():
        output = model(torch.zeros(2, 1, 28, 28))
    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()
