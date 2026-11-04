import json
import numpy as np
from utils import build_comparison
from utils.classification_metrics import calculate


def test_report_includes_available_current_experiment(tmp_path, monkeypatch):
    root=tmp_path/'mnist'; results=root/'results'; output=results/'comparison'
    path=results/'cnn_tssl_cnn_aligned_seed42'/'metrics.json'
    path.parent.mkdir(parents=True)
    labels=np.arange(10)
    path.write_text(json.dumps({'test':calculate(labels,np.eye(10))}),encoding='utf-8')
    monkeypatch.setattr(build_comparison,'ROOT',root)
    monkeypatch.setattr(build_comparison,'OUT',output)
    build_comparison.main()
    report=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    assert report['available'][0]['model']=='cnn_tssl_aligned'
    assert report['available'][0]['test_accuracy']==1.0
    assert 'tssl_repro' in report['missing']
