"""Export the complete frozen human-part confirmation and target quality tables."""
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/correspondence_reform_20260913'
PAPER = ROOT / 'paper'
METHODS = [('source', 'Source reference'), ('geometry', 'Geometry'),
           ('geometry_gain', 'Geometry + gains'), ('native', 'Member relation'),
           ('raw', 'Source-direction readout')]
QUERIES = ['full', 'pronouns', 'names', 'associated_words',
           'pronouns+names', 'pronouns+associated_words', 'names+associated_words']


def write_table(name, lines):
    (PAPER / 'tables' / (name + '.tex')).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    path = ART / 'R59_CONFIRMATION_ANALYSIS.json'
    d = json.loads(path.read_text())
    (PAPER / 'data/human_part_predictions_confirmation.json').write_text(json.dumps(d, indent=2) + '\n')
    cohort = d['new_target_seed_cohort']
    # Main table places use and fidelity side by side on the independent panel.
    main_lines = [r'\begin{tabular}{lrrr}\toprule',
                  r'\multicolumn{4}{l}{(a) Mean over annotated parts}\\\addlinespace',
                  r' & Profession & Part & Worst\\',
                  r'Method & accuracy & prediction & group\\\midrule']
    for key, label in METHODS:
        values = [cohort[key + '/parts'][metric]['value'] * 100
                  for metric in ['profession', 'balanced_agreement', 'worst_group']]
        main_lines.append(label.replace('Source-direction readout', r'Source-dir.\ readout') +
                          ' & ' + ' & '.join(f'{x:.2f}' for x in values) + r'\\')
    main_lines.extend([r'\midrule',
                       r'\multicolumn{4}{l}{(b) Profession accuracy after deleting each part}\\\addlinespace',
                       r'Method & Pronouns & Names & Words\\\midrule'])
    part_profiles = {}
    for method, label in METHODS:
        part_profiles[method] = {
            query: float(np.mean([
                d['per_target_seed'][str(seed)]['cells'][method + '/' + query]['profession']['value']
                for seed in range(2, 6)]))
            for query in ['pronouns', 'names', 'associated_words']}
        values = [100 * part_profiles[method][q] for q in part_profiles[method]]
        main_lines.append(label.replace('Source-direction readout', r'Source-dir.\ readout') +
                          ' & ' + ' & '.join(f'{x:.2f}' for x in values) + r'\\')
    main_lines.append(r'\bottomrule\end{tabular}')
    profile_export = {
        'source': str(path.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'target_seeds': [2, 3, 4, 5],
        'profession_accuracy': part_profiles,
        'scope': 'Descriptive means of all three predefined parts and all methods from the frozen R59 results; no new model calls, statistical test, selection or confirmation claim.'}
    (PAPER / 'data/human_part_profiles.json').write_text(
        json.dumps(profile_export, indent=2) + '\n', encoding='utf-8')
    write_table('human_part_confirmation_main', main_lines)
    # Every individual target, query and method is retained, including seed1.
    lines = [r'\begin{longtable}{llrrrrr}', r'\toprule',
             r'Target & Request & Source & Geometry & Gains & Relation & Readout\\\midrule\endhead']
    for seed in range(1, 6):
        cells = d['per_target_seed'][str(seed)]['cells']
        for query in QUERIES:
            values = [cells[m + '/' + query]['balanced_agreement']['value'] * 100 for m, _ in METHODS]
            lines.append(f'{seed} & {query.replace("associated_words", "words").replace("+", " + ")} & ' +
                         ' & '.join(f'{x:.2f}' for x in values) + r'\\')
        if seed < 5:
            lines.append(r'\addlinespace')
    lines.append(r'\bottomrule\end{longtable}')
    write_table('human_part_confirmation_all', lines)
    lines = [r'\begin{tabular}{llrrrr}\toprule',
             r'Requests & Method & Agreement & Prob.\ MAE & Profession & Worst group\\\midrule']
    for family in ['parts', 'unions', 'full']:
        for method, label in METHODS:
            values = [cohort[method + '/' + family][metric]['value'] * 100
                      for metric in ['balanced_agreement', 'probability_mae', 'profession', 'worst_group']]
            lines.append(f'{family.title()} & {label} & ' + ' & '.join(f'{x:.2f}' for x in values) + r'\\')
        if family != 'full':
            lines.append(r'\addlinespace')
    lines.append(r'\bottomrule\end{tabular}')
    write_table('human_part_confirmation_summary', lines)
    lines = [r'\begin{tabular}{llrrr}\toprule',
             r'Requests & Relation minus & Difference & Documents & Documents + targets\\\midrule']
    for family in ['parts', 'unions', 'full']:
        for control, label in [('geometry_gain', 'Geometry + gains'), ('geometry', 'Geometry'), ('raw', 'Readout')]:
            x = d['contrasts']['native-' + control + '/' + family]['balanced_agreement']
            def ci(k):
                lo, hi = x[k]
                return f'[{lo*100:.2f},{hi*100:.2f}]'
            lines.append(f'{family.title()} & {label} & {x["difference"]*100:.2f} & {ci("document_ci")} & {ci("document_target_seed_ci")}'+r'\\')
        if family != 'full':
            lines.append(r'\addlinespace')
    lines.append(r'\bottomrule\end{tabular}')
    write_table('human_part_confirmation_intervals', lines)
    run = ROOT / 'runs/REFORM_R59_shift_dictionaries_seeds2to5_v1_20260915'
    quality = [json.loads(line) for line in (run / 'metrics.raw.jsonl').read_text().splitlines()]
    assert len(quality) == 44 and all(r['kind'] == 'quality' for r in quality)
    lines = [r'\begin{tabular}{lrrrrr}\toprule',
             r'Site & FVE (\%) & CE recovery (\%) & $L_0$ & Active & Inactive\\\midrule']
    for site in dict.fromkeys(r['task'] for r in quality):
        rows = [r for r in quality if r['task'] == site]
        assert sorted(r['seed'] for r in rows) == [2, 3, 4, 5]
        def span(key, factor=1):
            v = np.array([r[key] for r in rows]) * factor
            return f'{v.mean():.2f} [{v.min():.2f},{v.max():.2f}]'
        fields = [site.replace('_', r'\_'), span('fve', 100), span('ce_recovery', 100),
                  f'{np.mean([r["l0"] for r in rows]):.2f}',
                  f'{min(r["alive"] for r in rows)}--{max(r["alive"] for r in rows)}',
                  f'{min(r["dead"] for r in rows)}--{max(r["dead"] for r in rows)}']
        lines.append(' & '.join(fields) + r'\\')
    lines.append(r'\bottomrule\end{tabular}')
    write_table('human_confirmation_material', lines)
    (PAPER / 'data/human_confirmation_material.json').write_text(json.dumps(dict(
        rows=quality, source=str(run.relative_to(ROOT)), raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest(),
        scope='Same final training recipe, four independent target initializations. Range describes targets, not a confidence interval. '
              'Active/inactive counts cover the2048-token validation panel and do not establish permanently dead features.'), indent=2) + '\n')
    print('Exported confirmation, all targets/queries, intervals and44 quality records.')


if __name__ == '__main__':
    main()
