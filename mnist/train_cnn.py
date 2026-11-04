"""Train an M5 CNN implementation under our common baseline data split."""
import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from data_pipeline.mnist import mnist_loaders

ROOT=Path(__file__).resolve().parent
from models.m5 import ModelM5
from models.tssl_cnn import TSSLCNN
from utils.classification_metrics import calculate
from utils.ema import ParameterEMA

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--architecture',choices=('m5','m5_small','tssl_cnn'),default='m5')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=120)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--augment',action='store_true')
    parser.add_argument('--ema',action='store_true')
    parser.add_argument('--tag',default='')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-batches',type=int,default=0,help='smoke test only')
    args=parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if args.tag and not all(c.isalnum() or c in '-_' for c in args.tag): raise ValueError('invalid tag')
    if (args.augment or args.ema or args.max_batches or args.architecture!='m5') and not args.tag: raise ValueError('use --tag to preserve previous results')
    loaders=mnist_loaders(args.batch_size,args.seed,normalization=((0.1307,),(0.3081,)) if args.architecture=='tssl_cnn' else None,augmentation=args.augment)
    model=(ModelM5() if args.architecture=='m5' else ModelM5((1,14,30,37,48,69)) if args.architecture=='m5_small' else TSSLCNN()).to(device)
    opt=(torch.optim.AdamW if args.architecture=='tssl_cnn' else torch.optim.Adam)(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=1.0 if args.architecture=='tssl_cnn' else .98)
    ema=ParameterEMA(model) if args.ema else None
    out=ROOT/'results'/('cnn_'+args.architecture+'_'+args.tag if args.tag else 'cnn_m5_kair'); out.mkdir(parents=True,exist_ok=True)
    def evaluate(loader,collect=False):
        model.eval(); n=correct=0; ys=[]; scores=[]
        with torch.no_grad():
            for i,(x,y) in enumerate(loader):
                y=y.to(device); v=model(x.to(device)); n+=len(y); correct+=(v.argmax(1)==y).sum().item()
                if collect: ys.extend(y.cpu().tolist()); scores.extend(v.cpu().tolist())
                if args.max_batches and i+1>=args.max_batches: break
        return calculate(ys,scores) if collect else {'accuracy':correct/n,'count':n}
    best=-1.; history=[]; start=time.time(); start_epoch=0
    if args.resume:
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if state['config'].get('architecture','m5')!=args.architecture: raise ValueError('resume architecture differs')
        for key in ('batch_size','lr','seed','augment','ema','max_batches'):
            if state['config'][key]!=getattr(args,key): raise ValueError('resume configuration differs: '+key)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        if ema: ema.shadow={k:v.to(device) for k,v in state['ema_shadow'].items()}; ema.updates=state['ema_updates']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng']);
        if device.type=='cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
        best=state['best']; history=state['history']; start_epoch=state['epoch']
    for e in range(start_epoch,args.epochs):
        model.train(); n=0; total=0.
        for i,(x,y) in enumerate(loaders[0]):
            x,y=x.to(device),y.to(device); opt.zero_grad(); loss=torch.nn.functional.nll_loss(model(x),y); loss.backward()
            if args.architecture=='tssl_cnn': torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            opt.step()
            if args.architecture=='tssl_cnn':
                with torch.no_grad():
                    for p in model.parameters(): p.clamp_(-4,4)
            if ema: ema.update(model)
            n+=len(y); total+=loss.item()*len(y)
            if args.max_batches and i+1>=args.max_batches: break
        with ema.apply(model) if ema else nullcontext():
            validation=evaluate(loaders[1]); history.append({'epoch':e+1,'train_loss':total/n,'validation_accuracy':validation['accuracy']})
            if validation['accuracy']>best:
                best=validation['accuracy']; torch.save({'model':model.state_dict(),'epoch':e+1,'args':vars(args)},out/'best.pt')
        (out/'progress.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(f"epoch={e+1} train_loss={total/n:.4f} val_acc={validation['accuracy']:.4f}",flush=True)
        scheduler.step()
        state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':e+1,'best':best,'history':history,'config':vars(args),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'ema_shadow':ema.shadow if ema else None,'ema_updates':ema.updates if ema else 0}
        temp=out/'latest.tmp'; torch.save(state,temp); temp.replace(out/'latest.pt')
    ck=torch.load(out/'best.pt',map_location=device,weights_only=True); model.load_state_dict(ck['model'])
    metrics=evaluate(loaders[2],collect=True)
    result={'model':args.architecture,'parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),'config':vars(args),'data_augmentation':args.augment,'ema':args.ema,'selection':'validation accuracy','best_epoch':ck['epoch'],'validation_best_accuracy':best,'test':metrics,'history':history,'runtime_seconds':time.time()-start,'torch':torch.__version__,'device':str(device),'note':'Independent implementation and common-split adaptation; not an exact paper reproduction.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('TEST',json.dumps({'accuracy':metrics['accuracy'],'macro_auc_ovr':metrics['macro_auc_ovr'],'count':metrics['count']}),flush=True)

if __name__=='__main__': main()
