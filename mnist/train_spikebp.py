"""Train the PyTorch-compatible Lee et al. implementation with a validation split."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from utils.classification_metrics import calculate
from data_pipeline.mnist import mnist_loaders

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'models'/'spikebp'))
import mnist_LeNet as legacy

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=125)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-batches',type=int,default=0)
    p.add_argument('--resume',action='store_true',help='resume from the latest completed epoch')
    p.add_argument('--tag',default='',help='optional output suffix')
    a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required by the author model')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    loaders=mnist_loaders(a.batch_size,a.seed)
    net=legacy.CNNModel().cuda(); opt=torch.optim.SGD(net.parameters(),lr=.0085,weight_decay=5e-4)
    if a.tag and not all(c.isalnum() or c in '-_' for c in a.tag): raise ValueError('invalid tag')
    out=ROOT/'results'/(('spikebp_smoke' if a.max_batches else 'spikebp')+('_'+a.tag if a.tag else '')); out.mkdir(parents=True,exist_ok=True)
    def run(loader,training,collect=False):
        net.train(training); n=correct=0; all_y=[]; all_scores=[]
        for i,(x,y) in enumerate(loader):
            x,y=x.cuda(),y.cuda()
            if training:
                opt.zero_grad(); result=net(x,steps=a.steps,l=.99); scores=result[0]
                grads=[legacy.grad_cal(.99,result[k],result[k-1]) for k in (2,4,6)]
                for activity,grad in zip(result[7:10],grads):
                    for spike in activity: spike.register_hook(lambda g, ng=grad:g*ng)
                target=torch.nn.functional.one_hot(y,10).to(scores.dtype)
                loss=torch.nn.functional.mse_loss(scores,target,reduction='sum'); loss.backward(); opt.step()
            else:
                with torch.no_grad(): scores=net.tst(x,steps=a.steps,l=.99)
            n+=len(y); correct+=(scores.argmax(1)==y).sum().item()
            if collect: all_y.extend(y.cpu().tolist()); all_scores.extend(scores.detach().cpu().tolist())
            if a.max_batches and i+1>=a.max_batches: break
        return calculate(all_y,all_scores) if collect else {'accuracy':correct/n,'count':n}
    best=-1; hist=[]; start_epoch=0
    latest=out/'latest.pt'
    if a.resume:
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if any(state['config'][k]!=getattr(a,k) for k in ('batch_size','steps','seed','max_batches')): raise ValueError('resume configuration differs')
        net.load_state_dict(state['net']); opt.load_state_dict(state['optimizer']); best=state['best']; hist=state['history']; start_epoch=state['epoch']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'Resuming after epoch {start_epoch}',flush=True)
    for e in range(start_epoch,a.epochs):
        # Match the original learning-rate schedule.
        lr=.0085*(.2 if e >= 75 else 1.)
        for group in opt.param_groups: group['lr']=lr
        tr=run(loaders[0],True); val_metrics=run(loaders[1],False)
        hist.append({'epoch':e+1,'train':tr,'validation':val_metrics,'lr':lr})
        if val_metrics['accuracy']>best:
            best=val_metrics['accuracy']; torch.save({'net':net.state_dict(),'epoch':e+1},out/'best.pt')
        state={'net':net.state_dict(),'optimizer':opt.state_dict(),'epoch':e+1,'best':best,'history':hist,'config':vars(a),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        temp=out/'latest.tmp'; torch.save(state,temp); temp.replace(latest)
        (out/'progress.json').write_text(json.dumps(hist,indent=2),encoding='utf-8')
        print(f"epoch={e+1} train={tr['accuracy']:.4f} val={val_metrics['accuracy']:.4f}",flush=True)
    ck=torch.load(out/'best.pt',weights_only=True); net.load_state_dict(ck['net']); result=run(loaders[2],False,collect=True)
    (out/'metrics.json').write_text(json.dumps({'config':vars(a),'test':result,'history':hist,'best_epoch':ck['epoch'],'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'note':'compatibility adaptation, validation split; not exact original training protocol'},indent=2),encoding='utf-8')
    print('TEST',result,flush=True)

if __name__=='__main__': main()
