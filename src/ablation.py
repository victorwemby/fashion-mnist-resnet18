# Run controlled experiments for augmentation and backbone freezing.
from __future__ import annotations
import argparse, csv, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--runs',type=int,default=1); p.add_argument('--output-dir',default='artifacts/ablation'); a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    configs=[('full_finetuning',[]),('no_augmentation',['--no-augmentation']),('frozen_backbone',['--freeze-backbone'])]; rows=[]
    for name,flags in configs:
        run_dir=out/name; cmd=[sys.executable,'src/train.py','--epochs',str(a.runs),'--batch-size','64','--output-dir',str(run_dir)]+flags; print('RUN',' '.join(cmd)); subprocess.run(cmd,check=True); history=(run_dir/'history.csv').read_text(encoding='utf-8').splitlines(); last=history[-1].split(','); rows.append({'setting':name,'val_accuracy':last[4],'val_macro_f1':last[5]})
    with (out/'ablation_results.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
if __name__=='__main__': main()

