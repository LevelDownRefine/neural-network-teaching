"""Train an M5 CNN implementation under our common baseline data split."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from utils.common_data import mnist_loaders

ROOT=Path(__file__).resolve().parent
from models.m5 import ModelM5
from utils.classification_metrics import calculate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=120)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders=mnist_loaders(args.batch_size,args.seed)
    model=ModelM5().to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=.98)
    out=ROOT/'results'/'cnn_m5_kair'; out.mkdir(parents=True,exist_ok=True)
    def evaluate(loader,collect=False):
        model.eval(); n=correct=0; ys=[]; scores=[]
        with torch.no_grad():
            for x,y in loader:
                y=y.to(device); v=model(x.to(device)); n+=len(y); correct+=(v.argmax(1)==y).sum().item()
                if collect: ys.extend(y.cpu().tolist()); scores.extend(v.cpu().tolist())
        return calculate(ys,scores) if collect else {'accuracy':correct/n,'count':n}
    best=-1.; history=[]; start=time.time()
    for e in range(args.epochs):
        model.train(); n=0; total=0.
        for x,y in loaders[0]:
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=torch.nn.functional.nll_loss(model(x),y); loss.backward(); opt.step()
            n+=len(y); total+=loss.item()*len(y)
        validation=evaluate(loaders[1]); history.append({'epoch':e+1,'train_loss':total/n,'validation_accuracy':validation['accuracy']})
        if validation['accuracy']>best:
            best=validation['accuracy']; torch.save({'model':model.state_dict(),'epoch':e+1,'args':vars(args)},out/'best.pt')
        (out/'progress.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(f"epoch={e+1} train_loss={total/n:.4f} val_acc={validation['accuracy']:.4f}",flush=True)
        scheduler.step()
    ck=torch.load(out/'best.pt',map_location=device,weights_only=True); model.load_state_dict(ck['model'])
    metrics=evaluate(loaders[2],collect=True)
    result={'model':'single M5-style CNN','config':vars(args),'data_augmentation':False,'ema':False,'selection':'validation accuracy','best_epoch':ck['epoch'],'validation_best_accuracy':best,'test':metrics,'history':history,'runtime_seconds':time.time()-start,'torch':torch.__version__,'device':str(device),'note':'Independent architecture implementation; comparison adaptation, not reproduction of 99.91% ensemble.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('TEST',json.dumps({'accuracy':metrics['accuracy'],'macro_auc_ovr':metrics['macro_auc_ovr'],'count':metrics['count']}),flush=True)

if __name__=='__main__': main()
