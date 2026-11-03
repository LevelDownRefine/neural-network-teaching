"""Build a shared report from completed test evaluations without rerunning models."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results'/'comparison'
SOURCES=[('logistic','logistic_kair/metrics.json','test','logistic_kair/auc.json'),
         ('cnn_m5','cnn_m5_kair/metrics.json','test',None),
         ('tssl','tssl/classification.json',None,None),
         ('spikebp','spikebp/metrics.json','test',None)]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary=[]; classes=[]; missing=[]
    for model,filename,key,auc_file in SOURCES:
        path=ROOT/'results'/filename
        if not path.exists(): missing.append(model); continue
        payload=json.loads(path.read_text(encoding='utf-8')); metrics=payload[key] if key else payload
        cm=np.asarray(metrics['confusion_matrix'],dtype=np.int64)
        if cm.shape!=(10,10) or cm.sum()!=metrics['count']: raise ValueError(f'invalid confusion matrix: {model}')
        correct=int(np.trace(cm)); accuracy=correct/int(cm.sum())
        if abs(accuracy-metrics['accuracy'])>1e-12: raise ValueError(f'accuracy mismatch: {model}')
        auc=metrics.get('macro_auc_ovr'); per_auc=None
        if auc_file:
            auc_data=json.loads((ROOT/'results'/auc_file).read_text(encoding='utf-8'))
            auc=auc_data['macro_auc_ovr']; per_auc=auc_data['per_class_auc']
        rows=metrics['per_class']
        for i,row in enumerate(rows):
            classes.append({'model':model,'digit':row['digit'],'precision':row['precision'],'recall':row['recall'],'f1':row['f1'],'support':row['support'],'auc':per_auc[i] if per_auc is not None else row.get('auc')})
        summary.append({'model':model,'test_accuracy':accuracy,'test_error':1-accuracy,'correct':correct,'total':int(cm.sum()),'macro_precision':float(np.mean([r['precision'] for r in rows])),'macro_recall':float(np.mean([r['recall'] for r in rows])),'macro_f1':float(np.mean([r['f1'] for r in rows])),'macro_auc_ovr':auc,'source':filename})
        with (OUT/f'{model}_confusion.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f); writer.writerow(['true\\predicted',*range(10)])
            for i,row in enumerate(cm): writer.writerow([i,*row.tolist()])
    for name,rows in [('summary',summary),('per_class',classes)]:
        if rows:
            with (OUT/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (OUT/'manifest.json').write_text(json.dumps({'available':summary,'missing':missing,'note':'TSSL split seed 3; other models seed 42. AUC uses softmax scores; SNN score semantics differ. SpikeBP is pending until full test evaluation.'},indent=2),encoding='utf-8')
    print('available:', ', '.join(row['model'] for row in summary), '; pending:', ', '.join(missing))

if __name__=='__main__': main()
