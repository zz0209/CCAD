from pathlib import Path
import csv
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/reuse_generalization_20260921_round02'


def main():
    source = OUT/'CONFIRMATION_ANALYSIS.json'
    result = json.loads(source.read_text())
    records = []
    for seed, study in result['studies'].items():
        for method, endpoints in study['metrics'].items():
            for endpoint, families in endpoints.items():
                for family, value in families.items():
                    records.append(dict(target=int(seed), method=method, endpoint=endpoint,
                                        family=family, nrmse=value))
    csv_path = ROOT/'paper/data/finite_execution_confirmation.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Execution & Full & Participation & Members \\', r'\midrule']
    panels = [
        ('Fitted dictionary', [2], [('Geometric', 'geometric'), ('Fixed members', 'saved_fixed'),
            ('Selected members', 'saved_sparse'), ('Trained dictionary', 'program'), ('Source readout', 'readout')]),
        ('Two further dictionaries', [4, 5], [('Geometric', 'geometric'), ('Transferred gains', 'transfer_gain'),
            ('Transferred fields', 'transfer_sparse'), ('Trained dictionaries', 'program'), ('Source readout', 'readout')])]
    for index, (label, seeds, methods) in enumerate(panels):
        if index:
            lines.append(r'\midrule')
        lines.append(r'\multicolumn{4}{l}{\emph{'+label+r'}} \\')
        for label, method in methods:
            values = [sum(result['studies'][str(s)]['metrics'][method]['later'][f] for s in seeds)/len(seeds)
                      for f in ['full', 'participation', 'members']]
            lines.append(label+' & '+' & '.join(f'{value:.3f}' for value in values)+r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    table = ROOT/'paper/tables/finite_execution_confirmation.tex'
    table.write_text('\n'.join(lines)+'\n')
    index_path = ROOT/'paper/EVIDENCE_INDEX.json'
    index = json.loads(index_path.read_text())
    index['finite_execution_confirmation'] = dict(
        analysis=str(source.relative_to(ROOT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        freeze=str((OUT/'FINITE_CONFIRMATION_FREEZE.json').relative_to(ROOT)),
        development=str((OUT/'RESULTS_AND_CHECKS.json').relative_to(ROOT)),
        table=str(table.relative_to(ROOT)), data=str(csv_path.relative_to(ROOT)),
        evidence='128new biographies and27requests. Target2 saved execution and targets4/5 field transfer. Intervals condition on these dictionaries and the four later heads.')
    index['finite_execution_confirmation']['diagnostics'] = [
        dict(path=str((OUT/name).relative_to(ROOT)), sha256=hashlib.sha256((OUT/name).read_bytes()).hexdigest())
        for name in ['FIELD_REALIZATION_DIAGNOSTIC.json','FIELD_SELECTION_DEVELOPMENT.json','NORMALIZATION_DIAGNOSTIC.json']]
    index_path.write_text(json.dumps(index, indent=2)+'\n')
    print(json.dumps(dict(rows=len(records), table=str(table), data=str(csv_path))))


if __name__ == '__main__':
    main()
