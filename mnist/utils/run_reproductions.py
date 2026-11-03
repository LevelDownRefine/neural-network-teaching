"""Run the approved CNN and TSSL experiments sequentially, logging each one."""
import json
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parent.parent
status=root/'results'/'reproduction_status.json'
jobs=[('cnn', ['train_cnn.py','--epochs','150','--augment','--ema','--tag','repro_seed42']),
      ('tssl', ['train_tssl.py','--epochs','100','--seed','42','--tag','repro_seed42'])]
status.parent.mkdir(parents=True,exist_ok=True)
for name,args in jobs:
    log=root/'results'/f'{name}_repro.log'
    with log.open('w',encoding='utf-8') as output:
        process=subprocess.Popen([sys.executable,'-u',str(root/args[0]),*args[1:]],cwd=root.parent,stdout=output,stderr=subprocess.STDOUT)
        status.write_text(json.dumps({'state':'running','model':name,'pid':process.pid,'log':str(log)},indent=2),encoding='utf-8')
        code=process.wait()
    if code:
        status.write_text(json.dumps({'state':'failed','model':name,'exit_code':code,'log':str(log)},indent=2),encoding='utf-8')
        raise SystemExit(code)
status.write_text(json.dumps({'state':'complete','models':[name for name,_ in jobs]},indent=2),encoding='utf-8')
