"""Small, reproducible SHD event visualization and SNN classification experiment."""
import argparse
import json
import platform
import random
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import snntorch as snn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'shd_data'
OUT = ROOT / 'results'


def load(path, indices, steps, bin_ms):
    frames = np.zeros((len(indices), steps, 700), dtype=np.float32)
    with h5py.File(path, 'r') as f:
        labels = np.asarray(f['labels'][...], dtype=np.int64)[indices]
        for j, index in enumerate(indices):
            times = f['spikes/times'][int(index)]
            units = f['spikes/units'][int(index)]
            bins = np.floor(times * 1000 / bin_ms).astype(np.int64)
            keep = (bins >= 0) & (bins < steps) & (units >= 0) & (units < 700)
            frames[j, bins[keep], units[keep].astype(np.int64)] = 1
    return torch.from_numpy(frames), torch.from_numpy(labels)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(700, 128)
        self.lif = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(128, 20)

    def forward(self, x):
        membrane = self.lif.init_leaky()
        scores = x.new_zeros((x.shape[0], 20))
        spike_sum = x.new_zeros(())
        for t in range(x.shape[1]):
            spikes, membrane = self.lif(self.fc1(x[:, t]), membrane)
            scores = scores + self.fc2(spikes)
            spike_sum = spike_sum + spikes.detach().sum()
        return scores / x.shape[1], spike_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train-size', type=int, default=3000)
    parser.add_argument('--valid-size', type=int, default=500)
    parser.add_argument('--test-size', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--bin-ms', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.train_size, args.valid_size, args.test_size, args.batch_size, args.steps, args.bin_ms) <= 0:
        parser.error('all counts and bin width must be positive')
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = np.random.default_rng(args.seed)
    with h5py.File(DATA/'shd_train.h5','r') as f: n_train=len(f['labels'])
    with h5py.File(DATA/'shd_test.h5','r') as f: n_test=len(f['labels'])
    if args.train_size + args.valid_size > n_train or args.test_size > n_test:
        raise ValueError('requested subset exceeds available samples')
    order = rng.permutation(n_train)
    train_idx = order[:args.train_size]
    valid_idx = order[args.train_size:args.train_size+args.valid_size]
    test_idx = rng.permutation(n_test)[:args.test_size]
    np.savez_compressed(OUT/'split_indices.npz', train=train_idx, validation=valid_idx, test=test_idx)
    datasets = [load(DATA/'shd_train.h5', train_idx, args.steps, args.bin_ms),
                load(DATA/'shd_train.h5', valid_idx, args.steps, args.bin_ms),
                load(DATA/'shd_test.h5', test_idx, args.steps, args.bin_ms)]
    print('Loaded subsets', [len(y) for _, y in datasets], flush=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    with h5py.File(DATA/'shd_train.h5','r') as f:
        for ax,index in zip(axes, train_idx[:3]):
            times=f['spikes/times'][int(index)]; units=f['spikes/units'][int(index)]
            ax.scatter(times*1000, units, s=1, rasterized=True)
            ax.set(ylabel='Channel', title=f"Training sample {index}, label {int(f['labels'][int(index)])}")
    axes[-1].set_xlabel('Time (ms)'); fig.tight_layout(); fig.savefig(OUT/'event_raster.png',dpi=150); plt.close(fig)
    train_loader = DataLoader(TensorDataset(*datasets[0]), batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    valid_loader = DataLoader(TensorDataset(*datasets[1]),batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(*datasets[2]),batch_size=args.batch_size)
    model=Classifier(); optim=torch.optim.Adam(model.parameters(),lr=0.001); loss_fn=nn.CrossEntropyLoss()

    def evaluate(loader):
        model.eval(); correct=total=spikes=inputs=0
        with torch.no_grad():
            for x,y in loader:
                scores, count=model(x); correct+=(scores.argmax(1)==y).sum().item()
                total+=len(y); spikes+=count.item(); inputs+=x.sum().item()
        return {'accuracy':correct/total,'correct':correct,'samples':total,
                'mean_input_spikes_per_sample':inputs/total,'mean_hidden_spikes_per_sample':spikes/total}

    best=-1; history=[]
    for epoch in range(args.epochs):
        model.train(); loss_total=correct=seen=0
        for x,y in train_loader:
            optim.zero_grad(set_to_none=True); scores,_=model(x); loss=loss_fn(scores,y)
            loss.backward(); optim.step()
            loss_total+=loss.item()*len(y); correct+=(scores.argmax(1)==y).sum().item(); seen+=len(y)
        validation=evaluate(valid_loader)
        entry={'epoch':epoch+1,'train_loss':loss_total/seen,'train_accuracy':correct/seen,'validation':validation}
        history.append(entry)
        if validation['accuracy'] > best:
            best=validation['accuracy']; torch.save({'model':model.state_dict(),'epoch':epoch+1,'config':vars(args)},OUT/'best.pt')
        print(f"epoch {epoch+1}: loss={loss_total/seen:.4f}, train={correct/seen:.3f}, validation={validation['accuracy']:.3f}",flush=True)
    checkpoint=torch.load(OUT/'best.pt', map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model']); test=evaluate(test_loader)
    result={'config':vars(args),'dataset_sizes':{'train':n_train,'test':n_test},'history':history,
            'best_epoch':checkpoint['epoch'],'test':test,'device':'cpu','python':platform.python_version(),
            'torch':str(torch.__version__),'snntorch':snn.__version__, 'numpy':np.__version__, 'h5py':h5py.__version__,
            'encoding':'binary event presence per channel and time bin; events beyond the window are discarded'}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot([r['epoch'] for r in history],[r['train_accuracy'] for r in history],label='Train'); ax.plot([r['epoch'] for r in history],[r['validation']['accuracy'] for r in history],label='Validation'); ax.set(xlabel='Epoch',ylabel='Accuracy',ylim=(0,1)); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT/'accuracy.png',dpi=160); plt.close(fig)
    print('TEST',json.dumps(test),flush=True)


if __name__ == '__main__': main()
