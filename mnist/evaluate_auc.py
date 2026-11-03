"""Compute one-vs-rest ROC AUC from saved baseline weights and full test set."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

root=Path(__file__).resolve().parent
name=sys.argv[1] if len(sys.argv)>1 else 'logistic'
folder=root/'results'/name
ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
model=nn.Sequential(nn.Flatten(),nn.Linear(784,10)) if ck['args'].get('model','logistic')=='logistic' else nn.Sequential(nn.Flatten(),nn.Linear(784,ck['args']['hidden']),nn.ReLU(),nn.Linear(ck['args']['hidden'],10))
model.load_state_dict(ck['model']); model.eval()
loader=DataLoader(datasets.MNIST(root/'data',train=False,download=False,transform=transforms.ToTensor()),batch_size=512)
scores=[]; labels=[]
with torch.no_grad():
    for x,y in loader:
        scores.append(torch.softmax(model(x),dim=1).numpy()); labels.append(y.numpy())
scores=np.concatenate(scores); labels=np.concatenate(labels)
def auc(y,s):
    order=np.argsort(s,kind='mergesort'); sorted_scores=s[order]
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_scores)!=0)+1]; ends=np.r_[starts[1:],len(s)]
    ranks=np.empty(len(s));
    for a,b in zip(starts,ends): ranks[order[a:b]]=(a+1+b)/2
    positives=y.sum(); negatives=len(y)-positives
    return float((ranks[y].sum()-positives*(positives+1)/2)/(positives*negatives))
per=[auc(labels==i,scores[:,i]) for i in range(10)]
out={'macro_auc_ovr':float(np.mean(per)),'per_class_auc':per,'score':'softmax probability','count':len(labels)}
(folder/'auc.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(name,json.dumps(out),flush=True)
