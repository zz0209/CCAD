from pathlib import Path
import argparse
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'containment'
TAB = ROOT / 'paper' / 'tables'
DESCRIPTION = 'Appendix tables of the reader-coverage manuscript, written from saved containment analyses.'

READERS = [('original', 'Orig.'), ('composer_surgeon_o0', 'C/S\\,0'), ('composer_surgeon_o1', 'C/S\\,1'),
           ('model_software_engineer_o0', 'M/E\\,0'), ('model_software_engineer_o1', 'M/E\\,1'), ('gender', 'Gend.'),
           ('bio_next_token', 'Bio NT'), ('mass_pronoun', 'Pron.'), ('mass_female', 'Fem.'), ('mass_male', 'Male'),
           ('state', 'State')]
GEMMA_ROWS = [('hungarian_decoder', 'Hungarian matching'), ('readout', 'Re-applied source edit\\textsuperscript{\\dag}'),
              ('natural_sup_hungarian', 'Generic text, random'), ('part_sup_hungarian', 'Classifier, parts'),
              ('e4c_subset_sup_hungarian', 'Classifier, parts and random'),
              ('e8a_random_sup_hungarian', 'Classifier, random'),
              ('lw_subset_state_sup_hungarian', 'State (LW), parts and random'),
              ('lw_random_state_sup_hungarian', 'State (LW), random'),
              ('std_subset_state_sup_hungarian', 'State (std.), parts and random'),
              ('std_random_state_sup_hungarian', 'State (std.), random')]
PYTHIA_ROWS = [('hungarian_decoder', 'Hungarian matching'), ('geometry_gain', 'Calibrated matching'),
               ('one_to_one', 'One-to-one, parts, gain up to 4'), ('natural_relation', 'Generic text, random'),
               ('subset_relation', 'Classifier, random'), ('parts_relation', 'Classifier, parts'),
               ('parts_relation_cap4', 'Classifier, parts, cap 4'), ('subset_relation_state', 'State, random'),
               ('parts_relation_state', 'State, parts')]


def fmt(v):
    return f'{v:.3f}'.lstrip('0') if v < 1 else f'{v:.2f}'


def table(rows, values, cols, extra_cols):
    head = ' & '.join(['Correspondence'] + [c for _, c in cols] + extra_cols)
    lines = ['\\begin{tabular}{l' + 'r' * (len(cols) + len(extra_cols)) + '}', '\\toprule', head + ' \\\\', '\\midrule']
    best = np.nanargmin(values, 0)
    for i, (_, label) in enumerate(rows):
        cells = [f'\\textbf{{{fmt(v)}}}' if best[j] == i else fmt(v) for j, v in enumerate(values[i])]
        lines.append(' & '.join([label] + cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines) + '\n'


def gemma_tables():
    runs = json.loads((ART / 'FINAL_GEMMA_READERS.json').read_text())['runs']
    for key, run in [('w65k', 'CSM_FINAL_GEMMA2B_readers_w65k_v1_20261011'),
                     ('l0low16k', 'CSM_FINAL_GEMMA2B_readers_l0low16k_v1_20261011')]:
        pt = runs[run]['point']
        values = np.array([[pt[m][f'parts|{r}'] for r, _ in READERS] +
                           [pt[m]['parts|fineweb_next_token'], pt[m]['parts|E2']] for m, _ in GEMMA_ROWS])
        (TAB / f'rc_gemma_readers_{key}.tex').write_text(table(GEMMA_ROWS, values, READERS, ['Gen.', 'E2']))


def pythia_table():
    runs = json.loads((ART / 'EXP1_PYTHIA.json').read_text())['runs']
    values = np.array([[np.mean([r['point'][m][f'parts|{k}'] for r in runs.values()]) for k, _ in READERS] +
                       [np.mean([r['point'][m]['parts|E2'] for r in runs.values()])] for m, _ in PYTHIA_ROWS])
    (TAB / 'rc_pythia_readers.tex').write_text(table(PYTHIA_ROWS, values, READERS, ['E2']))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.parse_args()
    gemma_tables()
    pythia_table()


if __name__ == '__main__':
    main()
