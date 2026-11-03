"""Plot confusion matrices from saved full-test evaluations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
for model in ('logistic','mlp','logistic_kair','mlp_kair','cnn_m5_kair'):
    path=ROOT/'results'/model/'metrics.json'
    if not path.exists(): continue
    m=json.loads(path.read_text(encoding='utf-8'))
    matrix=np.array(m['test']['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True)
    im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1200)
    for y in range(10):
        for x in range(10): ax.text(x,y,str(matrix[y,x]),ha='center',va='center',fontsize=7,color='white' if matrix[y,x]>600 else 'black')
    ax.set(xticks=range(10),yticks=range(10),xlabel='Predicted digit',ylabel='True digit',title=f"{model}: test accuracy {m['test']['accuracy']:.2%}")
    fig.colorbar(im,ax=ax,label='Number of images')
    fig.savefig(path.parent/'confusion_matrix.png',dpi=150); plt.close(fig)
