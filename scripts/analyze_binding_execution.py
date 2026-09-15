"""Analyze the frozen native-execution confirmation and its planned controls."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, contextlib, hashlib, io, json
import numpy as np
from analyze_binding_components import analyze


def main(run, output):
    with contextlib.redirect_stdout(io.StringIO()):
        analyze(run, output)
    out = json.loads(output.read_text())
    arrays = np.load(output.with_suffix('.npz'))
    contexts = arrays['contexts'].tolist()
    ops = arrays['operations'].tolist()
    n = len(contexts)
    weights = np.random.default_rng(9412219).multinomial(n, np.full(n, 1/n), size=10000)
    methods = sorted({k.split('|')[1] for k in arrays.files if '|' in k})
    scores = {}
    for method in methods:
        keys = sorted(k for k in arrays.files if '|' in k and k.split('|')[1] == method)
        scores[method] = np.stack([arrays[k][..., 0].prod(-1) for k in keys], axis=-1)
    primary = 'synthesized_code_bank'
    contrasts = []
    for reference in methods:
        if reference == primary:
            continue
        delta = scores[primary] - scores[reference]
        for oi, operation in [(None, 'macro')] + list(enumerate(ops)):
            d = delta if oi is None else delta[:, :, oi]
            cluster = d.reshape(n, -1).mean(1)
            contrasts.append(dict(method=primary, reference=reference, operation=operation,
                difference_points=float(100*cluster.mean()),
                interval_points=(100*np.quantile(weights@cluster/n, [.025, .975])).tolist(),
                seed_differences_points=(100*d.reshape(-1, d.shape[-1]).mean(0)).tolist()))
    raw = [json.loads(s) for s in (run/'metrics.raw.jsonl').read_text().splitlines()]
    source = {(r['component'], r['task'], r['seed'], r['operation'], r['query']): r['answer']
        for r in raw if r['kind']=='intervention' and r['method']=='country64'}
    pairs = {}
    for r in raw:
        if r['kind'] != 'intervention':
            continue
        key = (r['component'], r['task'], r['seed'], r['operation'])
        pairs.setdefault((r['method'], *key), []).append(r['answer'] == source[(*key, r['query'])])
    fidelity = []
    for method in methods:
        values = {c: [] for c in contexts}
        for key, same in pairs.items():
            if key[0] == method:
                assert len(same) == 2
                values[key[1]].append(all(same))
        cluster = np.array([np.mean(values[c]) for c in contexts])
        fidelity.append(dict(method=method, exact_source_pair_agreement_points=float(cluster.mean()*100),
            interval_points=(100*np.quantile(weights@cluster/n, [.025, .975])).tolist()))
    relations = [json.loads(s) for s in (run/'binding_relations.jsonl').read_text().splitlines()]
    identity = json.loads((Path('artifacts/correspondence_reform_20260913')/'R42_CONFIRMATION_FREEZE.json').read_text())
    checked = []
    for item in identity['files']:
        actual = hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
        checked.append(dict(path=item['path'], actual_sha256=actual, matches_freeze=actual==item['sha256']))
    assert all(r['matches_freeze'] for r in checked)
    out.update(execution_contrasts=contrasts, source_pair_fidelity=fidelity,
        relation_diagnostics=relations, frozen_files_checked=checked,
        secondary_note='Exact answer-pair agreement includes source errors; functional success is the primary endpoint.',
        written_at_utc=datetime.now(timezone.utc).isoformat())
    output.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(dict(macro=[r for r in out['macro'] if r['operation']=='macro'],
        contrasts=[r for r in contrasts if r['operation']=='macro'], fidelity=fidelity), indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True);a=p.parse_args();main(a.run,a.output)
