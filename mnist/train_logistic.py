"""Multinomial logistic-regression baseline; independent of the papers' code."""
import argparse
import gzip
import json
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from data_pipeline.mnist import mnist_loaders
from utils.classification_metrics import calculate
from utils.checkpoints import prepare_output

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--tag', default='', help='optional separate results suffix')
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device)
    raw = ROOT / 'data' / 'MNIST' / 'raw'
    for gz in raw.glob('*.gz'):
        target = gz.with_suffix('')
        if not target.exists():
            with gzip.open(gz, 'rb') as src, target.open('wb') as dst: shutil.copyfileobj(src, dst)
    loaders = mnist_loaders(a.batch_size,a.seed)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loss_fn = nn.CrossEntropyLoss()
    if a.tag and not all(c.isalnum() or c in '-_' for c in a.tag): raise ValueError('invalid tag')
    out = ROOT / 'results' / ('logistic' + ('_' + a.tag if a.tag else ''))
    prepare_output(out,False,allow_resume=False)
    def evaluate(loader, collect=False):
        model.eval(); total = correct = 0; loss = 0.; labels=[]; all_scores=[]
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                scores = model(x); pred = scores.argmax(1)
                loss += loss_fn(scores, y).item() * len(y); total += len(y); correct += (pred == y).sum().item()
                if collect: labels.extend(y.cpu().tolist()); all_scores.extend(scores.cpu().tolist())
        result=calculate(labels,all_scores) if collect else {'accuracy':correct/total, 'count':total}
        result['loss']=loss/total
        return result
    history=[]; best=-1.; start=time.time()
    for epoch in range(a.epochs):
        model.train(); total=0; loss=0.
        for x,y in loaders[0]:
            x,y=x.to(device),y.to(device); opt.zero_grad(); v=loss_fn(model(x),y); v.backward(); opt.step()
            total += len(y); loss += v.item()*len(y)
        val=evaluate(loaders[1]); history.append({'epoch':epoch+1,'train_loss':loss/total,'validation_accuracy':val['accuracy'],'validation_loss':val['loss']})
        if val['accuracy'] > best:
            best=val['accuracy']; torch.save({'model':model.state_dict(),'epoch':epoch+1,'args':vars(a)}, out / 'best.pt')
        print(f"epoch={epoch+1} train_loss={loss/total:.4f} val_acc={val['accuracy']:.4f}",flush=True)
    ckpt=torch.load(out/'best.pt',map_location=device,weights_only=True); model.load_state_dict(ckpt['model'])
    metrics=evaluate(loaders[2],collect=True)
    result={'experiment':'multinomial logistic regression', 'config':vars(a), 'best_epoch':ckpt['epoch'],'validation_best_accuracy':best, 'test':metrics,'history':history,'runtime_seconds':time.time()-start,'python':platform.python_version(),'torch':torch.__version__,'device':str(device)}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('TEST',json.dumps({'accuracy':metrics['accuracy'],'count':metrics['count'],'best_epoch':ckpt['epoch']}),flush=True)

if __name__ == '__main__': main()
