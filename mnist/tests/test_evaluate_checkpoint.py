import json
import sys
import torch
from torch import nn
from utils import evaluate_checkpoint


def test_logistic_checkpoint_evaluation_uses_common_metrics(tmp_path, monkeypatch):
    weights=tmp_path/'best.pt'; output=tmp_path/'test.json'
    model=nn.Sequential(nn.Flatten(),nn.Linear(784,10))
    torch.save({'model':model.state_dict(),'epoch':3},weights)
    class FakeMNIST(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return 10
        def __getitem__(self, index): return torch.zeros(1,28,28),index
    monkeypatch.setattr(evaluate_checkpoint.datasets,'MNIST',FakeMNIST)
    monkeypatch.setattr(sys,'argv',['evaluate','--model','logistic','--weights',str(weights),'--device','cpu','--output',str(output)])
    evaluate_checkpoint.main()
    result=json.loads(output.read_text(encoding='utf-8'))
    assert result['best_epoch']==3
    assert result['test']['count']==10
    assert len(result['test']['confusion_matrix'])==10
    assert result['test']['macro_auc_ovr']==.5
