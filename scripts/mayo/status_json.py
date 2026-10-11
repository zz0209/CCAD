import json
import subprocess
from pathlib import Path

ROOT = Path('/mnt/nvme3-data/harvardhds/ccad-m364569')


def count_lines(path):
    if not path.exists():
        return 0
    with path.open('rb') as f:
        return sum(chunk.count(b'\n') for chunk in iter(lambda: f.read(1 << 22), b''))


def main():
    runs = {}
    statuses = list(ROOT.glob('runs/*/*/status.json')) + list(ROOT.glob('code/CCAD/runs/CSM_*/status.json'))
    for status in statuses:
        run = status.parent
        st = json.loads(status.read_text())
        progress = run / 'progress.json'
        stage, items = None, None
        if progress.exists():
            p = json.loads(progress.read_text())
            stage = ' / '.join(str(p.get(k)) for k in ['stage', 'method', 'query', 'step'] if p.get(k) is not None)
            # SAE training runs report completed and total steps; program training reports the step of its current arm.
            if st.get('status') == 'RUNNING' and 'total_steps' in p:
                items = dict(done=p['completed_steps'] if 'completed_steps' in p else p['step'], total=p['total_steps'])
        # Runs that report finished items carry them as done and total in progress.jsonl.
        history = run / 'progress.jsonl'
        if st.get('status') == 'RUNNING' and history.exists():
            rows = [json.loads(line) for line in history.read_text().splitlines() if '"total"' in line]
            if rows:
                items = dict(done=rows[-1]['done'], total=rows[-1]['total'])
        runs[run.name] = dict(status=st.get('status'), started=st.get('started_at_utc') or st.get('updated_utc'), ended=st.get('ended_at_utc'),
                              records=count_lines(run / 'metrics.raw.jsonl'), stage=stage, items=items)
    downloads = {}
    for item in sorted((ROOT / 'assets').iterdir()):
        if item.is_dir():
            files = [p for p in item.rglob('*') if p.is_file() and '.cache' not in p.parts]
            downloads[item.name] = dict(files=len(files), size=f'{sum(p.stat().st_size for p in files) / 2**30:.2f} GiB')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True).stdout
    gpus = [dict(index=int(a), util=int(b), mem=int(c)) for a, b, c in (line.split(', ') for line in gpu.strip().splitlines())]
    sessions = subprocess.run(['tmux', 'ls'], capture_output=True, text=True).stdout.strip().splitlines()
    print(json.dumps(dict(runs=runs, downloads=downloads, gpus=gpus,
                          sessions=[s.split(':')[0] for s in sessions if s.startswith('ccad')])))


if __name__ == '__main__':
    main()
