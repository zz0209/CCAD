from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/final_science_20260920_round03'
PAPER = ROOT / 'paper'


def identity(path):
    return dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    paths = [ART / 'CONSUMER_DEVELOPMENT_ANALYSIS.json',
             ART / 'CONSUMER_FROZEN_HEAD_ANALYSIS.json']
    retrained, frozen = [json.loads(p.read_text()) for p in paths]
    assert retrained['classifier'] == 'retrained' and frozen['classifier'] == 'frozen'
    assert retrained['target_seeds'] == frozen['target_seeds'] == [2]
    names = [('none', 'Unedited'), ('source', 'Source explanation'),
             ('native', 'Member relation'), ('geometry_gain', 'Calibrated geometry'),
             ('raw', 'Source-direction readout'),
             ('raw_reconstruction', 'Reconstruction readout'),
             ('input_tangent_budget', 'Input-dependent initial'),
             ('input_gain', 'Source-column calibration'),
             ('input_program', 'Input-dependent trained')]
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'& \multicolumn{3}{c}{Fresh head} & Fixed head \\',
             r'Method & Full & Parts & Parts WG & Parts \\', r'\midrule']
    rows = []
    for method, label in names:
        r = retrained['results'][method]
        f = frozen['results'][method]
        values = [r['full']['profession']['mean'], r['parts_mean']['profession']['mean'],
                  r['parts_mean']['worst_group']['mean'], f['parts_mean']['profession']['mean']]
        assert all(0 <= v <= 1 for v in values)
        lines.append(label + ' & ' + ' & '.join(f'{100*v:.2f}' for v in values) + r' \\')
        rows.append(dict(method=method, label=label, fresh_full=values[0], fresh_parts=values[1],
                         fresh_parts_worst_group=values[2], fixed_parts=values[3]))
    lines += [r'\bottomrule', r'\end{tabular}']
    table = PAPER / 'tables/trained_program_consumer.tex'
    table.write_text('\n'.join(lines)+'\n')
    data = PAPER / 'data/trained_program_consumer.json'
    sources = [identity(p) for p in paths] + [identity(ART / 'CONSUMER_DEVELOPMENT_DESIGN.json')]
    selected = {k: retrained['contrasts']['input_program minus '+k]['parts_mean']
                for k in ['input_tangent_budget', 'input_gain', 'native', 'raw_reconstruction']}
    data.write_text(json.dumps(dict(rows=rows, primary_contrasts=selected, sources=sources,
        scope='Previously exposed development cohort; one target and one head initialization. Paired biography uncertainty, fixed task pairs.'), indent=2)+'\n')
    provenance = dict(sources=sources, generator=identity(Path(__file__)), outputs=[identity(table), identity(data)])
    manifest_path = PAPER / 'data/DATA_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['trained_program_consumer'] = provenance
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    index_path = PAPER / 'EVIDENCE_INDEX.json'
    index = json.loads(index_path.read_text())
    claim_id = 'trained_program_consumer'
    index['claims'] = [c for c in index['claims'] if c['id'] != claim_id]
    index['claims'].append(dict(id=claim_id, paper='Appendix input-dependent program confirmation',
        result='Frozen trained programs are evaluated in the existing fresh-head consumer on the exposed development cohort.',
        evidence=sources+provenance['outputs'], statistics=retrained['inference']))
    if claim_id not in index['current_manuscript_claim_ids']:
        index['current_manuscript_claim_ids'].append(claim_id)
    index['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    index_path.write_text(json.dumps(index, indent=2)+'\n')
    (ART / 'CONSUMER_EXPORT.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(table=table.as_posix(), primary=selected)))


if __name__ == '__main__':
    main()
