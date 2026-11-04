"""Run original TSSL network/loss with isolated validation and a held-out test set."""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'models' / 'tssl'
sys.path.insert(0, str(SRC))
import numpy as np
import torch
import yaml
from torchvision import datasets, transforms
from data_pipeline.mnist import mnist_loaders
from utils.classification_metrics import calculate
from torch.utils.data import DataLoader, random_split
import cnns
import global_v as glv
import functions.loss_f as loss_f

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=50)
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--hidden-size', type=int, default=300)
    p.add_argument('--official-protocol', action='store_true', help='60k training images and test-set checkpoint selection, as in official code')
    p.add_argument('--max-batches', type=int, default=0, help='smoke test only')
    p.add_argument('--cpu-smoke', action='store_true', help='test original CUDA-hardcoded code on CPU; not a reproduction run')
    p.add_argument('--tag',default='')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.cpu_smoke:
        if not a.max_batches: raise ValueError('--cpu-smoke requires --max-batches')
        torch.Tensor.cuda=lambda self,*args,**kwargs:self
        torch.nn.Module.cuda=lambda self,*args,**kwargs:self
    elif not torch.cuda.is_available(): raise RuntimeError('TSSL source requires CUDA')
    device='cpu' if a.cpu_smoke else 'cuda'
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    cfg=yaml.safe_load((SRC/'Networks'/'MNIST_CNN.yaml').read_text()); nc=cfg['Network']; nc['batch_size']=a.batch_size
    cfg['Layers']['FC_1']['n_outputs']=a.hidden_size; cfg['Layers']['output']['n_inputs']=a.hidden_size
    glv.init(nc['n_steps'],nc['tau_s'])
    if a.official_protocol:
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset=datasets.MNIST(ROOT/'data',train=True,download=False,transform=transform)
        testset=datasets.MNIST(ROOT/'data',train=False,download=False,transform=transform)
        trainloader=DataLoader(trainset,batch_size=a.batch_size,shuffle=True,num_workers=0)
        testloader=DataLoader(testset,batch_size=a.batch_size,shuffle=False,num_workers=0)
        loaders=(trainloader,testloader,testloader)
    else:
        loaders=mnist_loaders(a.batch_size,a.seed,normalization=((0.1307,), (0.3081,)))
    net=cnns.Network(nc,cfg['Layers'],[1,28,28]).cuda(); err=loss_f.SpikeLoss(nc).cuda()
    opt=torch.optim.AdamW(net.get_parameters(),lr=nc['lr'],betas=(0.9,0.999)); steps=nc['n_steps']
    desired=torch.tensor([0,1,1,1,1],device=device).view(1,1,1,1,steps)
    desired=loss_f.psp(desired,nc).view(1,1,1,steps)
    if a.tag and not all(c.isalnum() or c in '-_' for c in a.tag): raise ValueError('invalid tag')
    if a.hidden_size != 300 and not a.tag: raise ValueError('use --tag to preserve previous results')
    if a.official_protocol and not a.tag: raise ValueError('use --tag for official protocol')
    out=ROOT/'results'/(('tssl_smoke' if a.max_batches else 'tssl')+('_'+a.tag if a.tag else '')); out.mkdir(parents=True,exist_ok=True)
    if not a.tag and not a.max_batches and (out/'best.pt').exists(): raise ValueError('use --tag to avoid overwriting existing checkpoint')
    def run(loader, epoch, training, collect=False):
        net.train() if training else net.eval(); n=correct=0; losses=0.; ys=[]; all_scores=[]
        for idx,(x,y) in enumerate(loader):
            x=x.cuda().unsqueeze(-1).repeat(1,1,1,1,steps); y=y.cuda()
            with torch.set_grad_enabled(training):
                pred=net.forward(x,epoch,training)
                if training:
                    target=torch.zeros((len(y),10,1,1,steps),device=device); target[torch.arange(len(y),device=device),y]=desired
                    loss=err.spike_kernel(pred,target,nc); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.get_parameters(),1); opt.step(); net.weight_clipper(); losses+=loss.item()
            scores=pred.sum(dim=4).squeeze(-1).squeeze(-1); correct+=(scores.argmax(1)==y).sum().item(); n+=len(y)
            if collect: ys.extend(y.cpu().tolist()); all_scores.extend(scores.cpu().tolist())
            if a.max_batches and idx+1>=a.max_batches: break
        return calculate(ys,all_scores) if collect else {'accuracy':correct/n,'count':n,'loss':losses/n if training else None}
    best=-1; history=[]; start_epoch=0
    if a.resume:
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if state['config'].get('hidden_size',300)!=a.hidden_size: raise ValueError('resume configuration differs: hidden_size')
        if state['config'].get('official_protocol',False)!=a.official_protocol: raise ValueError('resume protocol differs')
        for key in ('batch_size','seed','max_batches','cpu_smoke'):
            if state['config'][key]!=getattr(a,key): raise ValueError('resume configuration differs: '+key)
        net.load_state_dict(state['net']); opt.load_state_dict(state['optimizer']); best=state['best']; history=state['history']; start_epoch=state['epoch']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng'])
        if not a.cpu_smoke: torch.cuda.set_rng_state_all(state['cuda_rng'])
    for e in range(start_epoch,a.epochs):
        tr=run(loaders[0],e,True); va=run(loaders[1],e,False); history.append({'epoch':e+1,'train':tr,('test_selection' if a.official_protocol else 'validation'):va})
        if va['accuracy']>best or (a.official_protocol and va['accuracy']==best):
            best=va['accuracy']; torch.save({'net':net.state_dict(),'epoch':e+1},out/'best.pt')
        state={'net':net.state_dict(),'optimizer':opt.state_dict(),'epoch':e+1,'best':best,'history':history,'config':vars(a),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if not a.cpu_smoke else []}
        temp=out/'latest.tmp'; torch.save(state,temp); temp.replace(out/'latest.pt')
        (out/'progress.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(f"epoch={e+1} train={tr['accuracy']:.4f} {'test_selection' if a.official_protocol else 'val'}={va['accuracy']:.4f}",flush=True)
    ck=torch.load(out/'best.pt',weights_only=True); net.load_state_dict(ck['net']); te=run(loaders[2],ck['epoch'],False,collect=True)
    (out/'metrics.json').write_text(json.dumps({'config':vars(a),'parameter_count':sum(p.numel() for p in net.get_parameters() if p.requires_grad),'selection':'official test accuracy (test set used for selection)' if a.official_protocol else 'validation accuracy','test':te,'best_epoch':ck['epoch'],'history':history,'source':'TSSL-BP repository; network and loss unchanged except configurable hidden width','device':'CPU smoke only' if a.cpu_smoke else torch.cuda.get_device_name(0),'torch':torch.__version__},indent=2),encoding='utf-8')
    print('TEST',te,flush=True)

if __name__=='__main__': main()
