"""Run original TSSL network/loss with isolated validation and a held-out test set."""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'sources' / 'TSSL-BP-master'
sys.path.insert(0, str(SRC))
import numpy as np
import torch
import yaml
from torchvision import datasets, transforms
from common_data import mnist_loaders
from torch.utils.data import DataLoader, random_split
import cnns
import global_v as glv
import functions.loss_f as loss_f

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=50)
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--max-batches', type=int, default=0, help='smoke test only')
    p.add_argument('--cpu-smoke', action='store_true', help='test original CUDA-hardcoded code on CPU; not a reproduction run')
    a=p.parse_args()
    if a.cpu_smoke:
        if not a.max_batches: raise ValueError('--cpu-smoke requires --max-batches')
        torch.Tensor.cuda=lambda self,*args,**kwargs:self
        torch.nn.Module.cuda=lambda self,*args,**kwargs:self
    elif not torch.cuda.is_available(): raise RuntimeError('TSSL source requires CUDA')
    device='cpu' if a.cpu_smoke else 'cuda'
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    cfg=yaml.safe_load((SRC/'Networks'/'MNIST_CNN.yaml').read_text()); nc=cfg['Network']; nc['batch_size']=a.batch_size
    glv.init(nc['n_steps'],nc['tau_s'])
    loaders=mnist_loaders(a.batch_size,a.seed,normalization=((0.1307,), (0.3081,)))
    net=cnns.Network(nc,cfg['Layers'],[1,28,28]).cuda(); err=loss_f.SpikeLoss(nc).cuda()
    opt=torch.optim.AdamW(net.get_parameters(),lr=nc['lr'],betas=(0.9,0.999)); steps=nc['n_steps']
    desired=torch.tensor([0,1,1,1,1],device=device).view(1,1,1,1,steps)
    desired=loss_f.psp(desired,nc).view(1,1,1,steps)
    out=ROOT/'results'/('tssl_smoke' if a.max_batches else 'tssl'); out.mkdir(parents=True,exist_ok=True)
    def run(loader, epoch, training):
        net.train() if training else net.eval(); n=correct=0; losses=0.
        for idx,(x,y) in enumerate(loader):
            x=x.cuda().unsqueeze(-1).repeat(1,1,1,1,steps); y=y.cuda()
            with torch.set_grad_enabled(training):
                pred=net.forward(x,epoch,training)
                if training:
                    target=torch.zeros((len(y),10,1,1,steps),device=device); target[torch.arange(len(y),device=device),y]=desired
                    loss=err.spike_kernel(pred,target,nc); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.get_parameters(),1); opt.step(); net.weight_clipper(); losses+=loss.item()
            scores=pred.sum(dim=4).squeeze(-1).squeeze(-1); correct+=(scores.argmax(1)==y).sum().item(); n+=len(y)
            if a.max_batches and idx+1>=a.max_batches: break
        return {'accuracy':correct/n,'count':n,'loss':losses/n if training else None}
    best=-1; history=[]
    for e in range(a.epochs):
        tr=run(loaders[0],e,True); va=run(loaders[1],e,False); history.append({'epoch':e+1,'train':tr,'validation':va})
        if va['accuracy']>best:
            best=va['accuracy']; torch.save({'net':net.state_dict(),'epoch':e+1},out/'best.pt')
        print(f"epoch={e+1} train={tr['accuracy']:.4f} val={va['accuracy']:.4f}",flush=True)
    ck=torch.load(out/'best.pt',weights_only=True); net.load_state_dict(ck['net']); te=run(loaders[2],ck['epoch'],False)
    (out/'metrics.json').write_text(json.dumps({'config':vars(a),'test':te,'best_epoch':ck['epoch'],'history':history,'source':'TSSL-BP repository; network and loss unchanged','device':'CPU smoke only' if a.cpu_smoke else torch.cuda.get_device_name(0),'torch':torch.__version__},indent=2),encoding='utf-8')
    print('TEST',te,flush=True)

if __name__=='__main__': main()
