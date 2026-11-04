"""Compute one-vs-rest ROC AUC from saved baseline weights and full test set."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from utils.classification_metrics import calculate

root=Path(__file__).resolve().parent.parent
name=sys.argv[1] if len(sys.argv)>1 else 'logistic'
folder=root/'results'/name
ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
if ck['args'].get('model','logistic') != 'logistic': raise ValueError('only logistic regression checkpoints are supported')
model=nn.Sequential(nn.Flatten(),nn.Linear(784,10))
model.load_state_dict(ck['model']); model.eval()
loader=DataLoader(datasets.MNIST(root/'data',train=False,download=False,transform=transforms.ToTensor()),batch_size=512)
scores=[]; labels=[]
with torch.no_grad():
    for x,y in loader:
        scores.append(model(x).numpy()); labels.append(y.numpy())
scores=np.concatenate(scores); labels=np.concatenate(labels)
metrics=calculate(labels,scores)
out={'macro_auc_ovr':metrics['macro_auc_ovr'],'per_class_auc':[x['auc'] for x in metrics['per_class']],'score':'softmax probability','count':len(labels)}
(folder/'auc.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(name,json.dumps(out),flush=True)
