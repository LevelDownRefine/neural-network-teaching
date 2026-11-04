from pathlib import Path

import torch
from models.m5 import ModelM5
from models.tssl_cnn import TSSLCNN


def test_saved_cnn_checkpoint_loads_and_predicts():
    path = Path(__file__).resolve().parents[1] / "results" / "cnn_m5_kair" / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ModelM5().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.no_grad():
        output = model(torch.zeros(2, 1, 28, 28))
    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()
    assert torch.allclose(output.exp().sum(dim=1), torch.ones(2), atol=1e-6)


def test_tssl_cnn_shapes_and_parameter_count():
    model = TSSLCNN()
    assert sum(p.numel() for p in model.parameters()) == 210375
    output = model(torch.randn(2, 1, 28, 28))
    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()
    torch.nn.functional.nll_loss(output, torch.tensor([1, 2])).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
