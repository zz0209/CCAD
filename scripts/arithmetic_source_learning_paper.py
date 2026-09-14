"""Export the frozen equivalent-source experiment and its development history."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/correspondence_reform_20260913'
PAPER = ROOT / 'paper'


def main():
    confirmation = ART / 'r34_confirmation/confirmation.json'
    c = json.loads(confirmation.read_text())
    inventory, evidence = [], []
    for run in sorted((ROOT / 'runs').glob('REFORM_R34_*')):
        status = json.loads((run / 'status.json').read_text())
        assert status['status'] in ['PASS', 'FAIL', 'CUT']
        summary = json.loads((run / 'metrics.summary.json').read_text()) if (run / 'metrics.summary.json').exists() else None
        raw = run / 'metrics.raw.jsonl'
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        if summary is not None:
            assert digest == summary['metrics_raw_sha256']
        else:
            assert status['status'] == 'CUT' and raw.stat().st_size == 0
        files = ['config.resolved.json', 'status.json', 'code_hashes.json', 'inputs.json',
                 'environment.json', 'metrics.raw.jsonl', 'metrics.summary.json', 'contract_validation.json']
        files += [p.name for p in run.glob('*FREEZE.json')]
        inventory.append(dict(run=run.relative_to(ROOT).as_posix(), status=status,
                              summary=summary, raw_sha256=digest))
        evidence.extend((run / f).relative_to(ROOT).as_posix() for f in files if (run / f).exists())
    (ART / 'R34_RUNS.json').write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        runs=inventory, scope='Includes all preserved attempts, failures and diagnostics; run count is not a scientific result.'), indent=2)+'\n')

    labels = [('source_original', 'Original source'), ('source_views', 'Equivalent-input source'),
              ('direct_320', 'Direct gradient, 320 batches'), ('field_old', 'Original-source field'),
              ('field_views', 'Equivalent-source field'), ('assignment_views', 'Equivalent-source assignment')]
    lines = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'& \multicolumn{3}{c}{Units replacement} & \multicolumn{3}{c}{Tens replacement} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
             r'Method & H & T & P & H & T & P \\', r'\midrule']
    for method, label in labels:
        values = [100 * next(x for x in c['cells'] if x['initialization'] == method and x['operation'] == op)[m]
                  for op in ['unit', 'tens'] for m in ['exact_hybrid', 'target_digit_success', 'preserve_digit_success']]
        lines.append(label + ' & ' + ' & '.join(f'{v:.2f}' for v in values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (PAPER / 'tables/arithmetic_source_learning.tex').write_text('\n'.join(lines)+'\n')
    (PAPER / 'data/arithmetic_source_learning.json').write_text(json.dumps(c, indent=2)+'\n')
    # Small regularizer ablation is descriptive; all later corrected-view runs remain in the inventory.
    original = json.loads((ART / 'r34_latent_original.json').read_text())
    shifted = json.loads((ART / 'r34_latent_new_range.json').read_text())
    methods = list(original['mean_hybrid'])
    ablation = [dict(method=m, original=original['mean_hybrid'][m], shifted=shifted['mean_hybrid']['adapt_'+m+'_u0']) for m in methods]
    (PAPER / 'data/arithmetic_latent_ablation.json').write_text(json.dumps(ablation, indent=2)+'\n')
    evidence += ['artifacts/correspondence_reform_20260913/r34_confirmation/'+f for f in ['confirmation.json','cluster_outcomes.npz']]
    evidence += ['scripts/arithmetic_source_learning_paper.py', 'scripts/analyze_arithmetic_reuse_confirmation.py',
                 'scripts/plot_arithmetic_source_learning.py', 'scripts/arithmetic_counterfactual_fit.py',
                 'scripts/arithmetic_relation_transfer.py', 'scripts/run_arithmetic_digit_components.py',
                 'paper/data/arithmetic_source_learning.json', 'paper/data/arithmetic_latent_ablation.json',
                 'paper/tables/arithmetic_source_learning.tex', 'paper/sections/arithmetic_source_learning_results.tex',
                 'paper/figures/arithmetic_source_learning.pdf', 'artifacts/correspondence_reform_20260913/R34_RUNS.json']
    claim = dict(id='arithmetic_equivalent_source_transfer', paper='Source invariance and cross-seed functional transfer',
                 result=c['primary'], source_diagnostics=c['secondary'], scope=c['scope'], evidence=evidence)
    (PAPER / 'data/arithmetic_source_learning_evidence.json').write_text(json.dumps(claim, indent=2)+'\n')
    fp = PAPER / 'figures/FIGURE_MANIFEST.json'
    fm = json.loads(fp.read_text())
    figure_paths = ['figures/arithmetic_source_learning.'+ext for ext in ['pdf', 'svg', 'png']]
    fm['outputs'] = [x for x in fm['outputs'] if x['path'] not in figure_paths]
    fm['outputs'].extend(dict(path=p, bytes=(PAPER / p).stat().st_size,
                             sha256=hashlib.sha256((PAPER / p).read_bytes()).hexdigest()) for p in figure_paths)
    fm.setdefault('additional_sources', {})['arithmetic_source_learning'] = dict(
        path='data/arithmetic_source_learning.json',
        sha256=hashlib.sha256((PAPER / 'data/arithmetic_source_learning.json').read_bytes()).hexdigest(),
        generator='scripts/plot_arithmetic_source_learning.py')
    fp.write_text(json.dumps(fm, indent=2)+'\n')
    print(json.dumps(dict(runs=len(inventory), means={m:sum(x['exact_hybrid'] for x in c['cells'] if x['initialization']==m)/2 for m,_ in labels}, primary=c['primary'])))


if __name__ == '__main__':
    main()
