"""Regenerate the CCAD manuscript from retained result data and compile LaTeX.

Use --skip-data for a paper-only copy with its exported data. By default the
compiler uses its existing cache; --allow-download permits official TeX bundle
downloads. This script does not train, fit, infer, or publish anything.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper',type=Path,default=ROOT/'paper')
    parser.add_argument('--tectonic',type=Path,default=ROOT/'.aris/tex_runtime_v1/tectonic.exe')
    parser.add_argument('--skip-data',action='store_true')
    parser.add_argument('--skip-figures',action='store_true')
    parser.add_argument('--allow-download',action='store_true')
    args=parser.parse_args();paper=args.paper.resolve();compiler=args.tectonic.resolve()
    if not compiler.is_file():raise FileNotFoundError(f'Tectonic not found: {compiler}; pass --tectonic to the installed 0.17.0 executable')
    start=dt.datetime.now(dt.timezone.utc);timer=time.monotonic();steps=[]
    buildroot=paper/'build';buildroot.mkdir(exist_ok=True,parents=True)
    logdir=buildroot/start.strftime('%Y%m%dT%H%M%S%fZ');logdir.mkdir()
    env=os.environ.copy();env.setdefault('TECTONIC_CACHE_DIR',str(ROOT/'.aris/tex_runtime_v1/cache'))
    for name,cmd,cwd in [
        ('data',[sys.executable,str(ROOT/'scripts/build_paper_data.py'),'--output',str(paper)],ROOT),
        ('figures',[sys.executable,str(ROOT/'scripts/build_paper_figures.py'),'--paper',str(paper)],ROOT),
        ('latex',[str(compiler),'--untrusted','--keep-logs','--keep-intermediates']+([] if args.allow_download else ['--only-cached'])+['main.tex'],paper)]:
        if (name=='data' and args.skip_data) or (name=='figures' and args.skip_figures):continue
        t=time.monotonic()
        with (logdir/f'{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,text=True)
        steps.append(dict(step=name,returncode=result.returncode,wall_seconds=time.monotonic()-t,command=cmd,log=(logdir/f'{name}.log').relative_to(paper).as_posix()))
        print(json.dumps(steps[-1]),flush=True)
        if result.returncode:
            (logdir/'BUILD_FAILED.json').write_text(json.dumps(dict(started_at_utc=start.isoformat(),steps=steps),indent=2)+'\n')
            raise SystemExit(result.returncode)
    pdf=paper/'main.pdf'
    if not pdf.is_file():raise FileNotFoundError(pdf)
    files=[p for p in paper.rglob('*') if p.is_file() and p.suffix in ['.tex','.bib','.pdf','.csv','.json','.py','.md','.svg'] and 'build' not in p.relative_to(paper).parts]
    receipt=dict(started_at_utc=start.isoformat(),completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),wall_seconds=time.monotonic()-timer,steps=steps,
        pdf=dict(path=str(pdf),bytes=pdf.stat().st_size,sha256=hashlib.sha256(pdf.read_bytes()).hexdigest()),
        files=[dict(path=p.relative_to(paper).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)],
        scope='Deterministic manuscript/data/figure build, not experiment replay or scientific validation.')
    (logdir/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    (buildroot/'BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt['pdf']))


if __name__=='__main__':
    main()
