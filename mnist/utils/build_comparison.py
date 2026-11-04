"""Build a shared report from completed test evaluations without rerunning models."""
import csv
import json
from pathlib import Path
import numpy as np
from utils.results import discover

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'comparison'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, classes = [], []
    for record in discover(ROOT / 'results'):
        name, metrics = record['name'], record['metrics']
        cm = np.asarray(metrics['confusion_matrix'], dtype=np.int64)
        if cm.shape != (10, 10) or cm.sum() != metrics['count']:
            raise ValueError(f'invalid confusion matrix: {name}')
        accuracy = int(np.trace(cm)) / int(cm.sum())
        if abs(accuracy - metrics['accuracy']) > 1e-12:
            raise ValueError(f'accuracy mismatch: {name}')
        rows = metrics['per_class']
        for row in rows:
            classes.append({'model': name, **{k: row.get(k) for k in ('digit', 'precision', 'recall', 'f1', 'support', 'auc')}})
        summary.append({'model': name, 'test_accuracy': accuracy, 'test_error': 1-accuracy,
                        'correct': int(np.trace(cm)), 'total': int(cm.sum()),
                        'macro_precision': float(np.mean([r['precision'] for r in rows])),
                        'macro_recall': float(np.mean([r['recall'] for r in rows])),
                        'macro_f1': float(np.mean([r['f1'] for r in rows])),
                        'macro_auc_ovr': metrics.get('macro_auc_ovr'),
                        'seed': record['seed'], 'selection': record['selection'], 'split': json.dumps(record['split']) if record['split'] else None,
                        'source': str(record['path'].relative_to(ROOT / 'results'))})
        with (OUT / f'{name}_confusion.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f); writer.writerow(['true\\predicted', *range(10)])
            for i, row in enumerate(cm): writer.writerow([i, *row.tolist()])
    for name, rows in [('summary', summary), ('per_class', classes)]:
        if rows:
            with (OUT / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (OUT / 'manifest.json').write_text(json.dumps({'available': summary, 'note': 'Check seed, selection and configuration before comparing experiments.'}, indent=2), encoding='utf-8')
    print('available:', ', '.join(row['model'] for row in summary))


if __name__ == '__main__': main()
