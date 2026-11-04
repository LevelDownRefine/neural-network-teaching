"""Evaluate the selected TSSL checkpoint on all official test images."""

import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from classification_metrics import calculate
from utils.io import atomic_write_json

root = Path(__file__).resolve().parent.parent
src = root / "models" / "tssl"
sys.path.insert(0, str(src))
import cnns
import global_v as glv

cfg = yaml.safe_load((src / "Networks" / "MNIST_CNN.yaml").read_text())
nc = cfg["Network"]
glv.init(nc["n_steps"], nc["tau_s"])
net = cnns.Network(nc, cfg["Layers"], [1, 28, 28]).cuda()
ck = torch.load(
    root / "results" / "tssl" / "best.pt", map_location="cuda", weights_only=True
)
net.load_state_dict(ck["net"])
net.eval()
tf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
loader = DataLoader(
    datasets.MNIST(root / "data", train=False, download=False, transform=tf),
    batch_size=50,
    shuffle=False,
    num_workers=0,
)
ys = []
scores = []
with torch.no_grad():
    for x, y in loader:
        x = x.cuda().unsqueeze(-1).repeat(1, 1, 1, 1, nc["n_steps"])
        output = net.forward(x, ck["epoch"] - 1, False)
        scores.append(output.sum(dim=4).squeeze(-1).squeeze(-1).cpu().numpy())
        ys.append(y.numpy())
metrics = calculate(np.concatenate(ys), np.concatenate(scores))
metrics["selected_epoch"] = ck["epoch"]
atomic_write_json(root / "results" / "tssl" / "classification.json", metrics)
print(
    "TSSL test accuracy",
    metrics["accuracy"],
    "macro AUC",
    metrics["macro_auc_ovr"],
    flush=True,
)
