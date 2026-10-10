import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
values = np.load(run / 'responses.npz')
panel = json.loads((run / 'panel.json').read_text())
queries = list(panel['query_order'])
tasks = np.array([r['task'] for r in panel['rows']])
none, source = values['none'], values['source']
for t in sorted(set(tasks)):
    cols = tasks == t
    base = none[0, cols]
    print(t, 'clean margin mean', round(float(base.mean()), 3), 'acc', round(float((base > 0).mean()), 3))
    for q in ['full', 'verb', 'number', 'gender']:
        eff = source[queries.index(q), cols] - base
        print('   source effect', q, 'rms', round(float(np.sqrt((eff ** 2).mean())), 4), 'mean', round(float(eff.mean()), 4),
              'acc after', round(float((source[queries.index(q), cols] > 0).mean()), 3))
progress = run / 'stdout.log'
lines = [json.loads(l) for l in progress.read_text().splitlines() if l.startswith('{') and '"TRAINING"' in l]
for method in sorted({l.get('method') for l in lines}):
    seq = [l for l in lines if l.get('method') == method]
    print(method, [(l.get('step'), round(l.get('program', float('nan')), 4), round(l.get('reconstruction', float('nan')), 4)) for l in seq[::4]][:12])
