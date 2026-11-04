"""MNIST validation-protocol adaptation of the original PLIF experiment.

Requires the original PLIF code and the SpikingJelly revision specified in README.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from utils.classification_metrics import calculate

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plif-source', required=True)
    p.add_argument('--sj-source', required=True)
    p.add_argument('--epochs', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=2020)
    p.add_argument('--tag', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-batches', type=int, default=0, help='smoke test only')
    a = p.parse_args()
    if not all(c.isalnum() or c in '-_' for c in a.tag): raise ValueError('invalid tag')
    sys.path.insert(0, str(Path(a.plif_source).resolve() / 'codes'))
    sys.path.insert(0, str(Path(a.sj_source).resolve()))
    import models
    from spikingjelly.clock_driven import functional
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    train_transform, test_transform = models.get_transforms('MNIST')
    origin = datasets.MNIST(ROOT/'data', train=True, transform=train_transform, download=False)
    test = datasets.MNIST(ROOT/'data', train=False, transform=test_transform, download=False)
    # The original enumerates transformed images while constructing a per-class split.
    grouped = [[] for _ in range(10)]
    for i in range(len(origin)):
        _, label = origin[i]
        grouped[label].append(i)
    train_idx, val_idx = [], []
    for indices in grouped:
        cut = math.ceil(len(indices) * 0.85)
        train_idx.extend(indices[:cut]); val_idx.extend(indices[cut:])
    train = DataLoader(Subset(origin, train_idx), batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val = DataLoader(Subset(origin, val_idx), batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=a.batch_size*8, shuffle=False, num_workers=0)
    net = models.MNISTNet(T=8,init_tau=2.,use_plif=True,use_max_pool=True,alpha_learnable=False,detach_reset=True).to(a.device)
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=64)
    out=ROOT/'results'/('plif_'+a.tag); out.mkdir(parents=True,exist_ok=True)
    best=-1.; history=[]; start=0
    if a.resume:
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        for key in ('batch_size','seed','max_batches'):
            if state['config'][key]!=getattr(a,key): raise ValueError('resume configuration differs: '+key)
        net.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        best=state['best']; history=state['history']; start=state['epoch']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda_rng'])
    def evaluate(loader, collect=False):
        net.eval(); correct=n=0; labels=[]; scores=[]
        with torch.no_grad():
            for i,(x,y) in enumerate(loader):
                x,y=x.to(a.device),y.to(a.device); s=net(x)/net.T
                correct+=(s.argmax(1)==y).sum().item(); n+=len(y)
                if collect: labels.extend(y.cpu().tolist()); scores.extend(s.cpu().tolist())
                functional.reset_net(net)
                if a.max_batches and i+1>=a.max_batches: break
        return calculate(labels,scores) if collect else correct/n
    print(f'train_samples={len(train_idx)} validation_samples={len(val_idx)} parameters={sum(p.numel() for p in net.parameters())}',flush=True)
    for e in range(start,a.epochs):
        net.train(); loss_sum=n=0
        for i,(x,y) in enumerate(train):
            x,y=x.to(a.device),y.to(a.device); optimizer.zero_grad()
            s=net(x)/net.T; loss=F.mse_loss(s,F.one_hot(y,10).float()); loss.backward(); optimizer.step(); functional.reset_net(net)
            loss_sum+=loss.item()*len(y); n+=len(y)
            if a.max_batches and i+1>=a.max_batches: break
        scheduler.step(); score=evaluate(val); history.append({'epoch':e+1,'train_loss':loss_sum/n,'validation_accuracy':score})
        if score>=best:
            best=score; torch.save({'model':net.state_dict(),'epoch':e+1},out/'best.pt')
        state={'model':net.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':e+1,'best':best,'history':history,'config':vars(a),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        temp=out/'latest.tmp'; torch.save(state,temp); temp.replace(out/'latest.pt')
        (out/'progress.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(f'epoch={e+1} loss={loss_sum/n:.5f} validation={score:.5f} best={best:.5f}',flush=True)
    ck=torch.load(out/'best.pt',map_location=a.device,weights_only=True); net.load_state_dict(ck['model'])
    metrics=evaluate(test_loader,collect=True)
    result={'config':vars(a),'best_epoch':ck['epoch'],'validation_best_accuracy':best,'selection':'validation accuracy','train_count':len(train_idx),'validation_count':len(val_idx),'parameter_count':sum(p.numel() for p in net.parameters()),'test':metrics,'history':history,'notes':'Author 15% stratified split, validation random augmentation/shuffle/drop_last; test evaluated once after selection; num_workers=0.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('TEST',json.dumps({'accuracy':metrics['accuracy'],'macro_auc_ovr':metrics['macro_auc_ovr'],'count':metrics['count']}),flush=True)


if __name__=='__main__': main()
