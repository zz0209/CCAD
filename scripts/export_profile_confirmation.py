from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.aris/plot_runtime_v1'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ART = ROOT / 'artifacts/final_science_20260920'
PAPER = ROOT / 'paper'
PROFILE = 'initial_refined_source_metric'
SETTINGS = ['human_later_heads', 'grammar', 'human_original_head']
METHODS = [('Source-direction readout', 'initial_raw_readout'),
           ('Active-region columns', 'initial_tangent'),
           ('Euclidean refinement', 'initial_refined_common'),
           ('Source-profile refinement', PROFILE)]


def identity(path):
    return dict(path=str(path.relative_to(ROOT)).replace('\\', '/'), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    source = ART / 'PROFILE_CONFIRMATION_ANALYSIS_V2.json'
    data = json.loads(source.read_text())
    assert data['complete_frozen_cohort']
    assert all(data[s]['seeds'] == [1, 2, 3, 4, 5] for s in SETTINGS)
    shutil.copyfile(source, PAPER / 'data/source_profile_confirmation.json')
    rows = []
    for setting in SETTINGS:
        for cohort, result in [('five_targets', data[setting]), ('trained_comparison', data[setting]['trained_reference'])]:
            for method, families in result['summary'].items():
                for family, values in families.items():
                    rows.append(dict(setting=setting, cohort=cohort, method=method, family=family,
                        nrmse=values['nrmse'], lower=values['interval'][0], upper=values['interval'][1],
                        seeds=json.dumps(result['seeds']), by_seed=json.dumps(values['by_seed']),
                        valid_bootstrap_draws=values['valid_bootstrap_draws']))
    with (PAPER / 'data/source_profile_confirmation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Execution & Later classifiers & Grammar & Original classifier \\', r'\midrule',
             r'\multicolumn{4}{l}{Five unchanged target dictionaries} \\']
    for label, method in METHODS:
        values = [data[s]['summary'][method]['participation']['nrmse'] for s in SETTINGS]
        table.append(label + ' & ' + ' & '.join(f'{v:.3f}' for v in values) + r' \\')
    table += [r'\midrule', r'\multicolumn{4}{l}{Matched four-target comparison with trained dictionaries} \\']
    for label, method in [('Source-profile refinement', PROFILE), ('Program-trained execution', 'task_adapted_tangent')]:
        values = [data[s]['trained_reference']['summary'][method]['participation']['nrmse'] for s in SETTINGS]
        table.append(label + ' & ' + ' & '.join(f'{v:.3f}' for v in values) + r' \\')
    table += [r'\bottomrule', r'\end{tabular}']
    (PAPER / 'tables/source_profile_confirmation.tex').write_text('\n'.join(table)+'\n')
    macros = []
    for setting, prefix in zip(SETTINGS, ['ProfileHuman', 'ProfileGrammar', 'ProfileOriginal']):
        for method, suffix in [('initial_refined_common', 'Euclidean'), (PROFILE, 'Weighted'),
                               ('initial_tangent', 'Active'), ('initial_raw_readout', 'Readout')]:
            value = data[setting]['summary'][method]['participation']['nrmse']
            macros.append('\\newcommand{\\'+prefix+suffix+'}{'+f'{value:.3f}'+'}')
        contrast = next(c for c in data[setting]['contrasts'] if c['family']=='participation' and c['reference']=='initial_refined_common')
        for suffix, value in [('Reduction', contrast['reduction']), ('Lower', contrast['interval'][0]), ('Upper', contrast['interval'][1])]:
            macros.append('\\newcommand{\\'+prefix+suffix+'}{'+f'{value:.3f}'+'}')
    (PAPER / 'tables/source_profile_values.tex').write_text('\n'.join(macros)+'\n')
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman', 'font.size':9, 'mathtext.fontset':'stix',
        'axes.titlesize':10, 'axes.labelsize':9, 'xtick.labelsize':8, 'ytick.labelsize':9,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.spines.left':False,
        'axes.linewidth':.6, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    labels = ['Source-direction readout', 'Active-region columns', 'Euclidean refinement', 'Program-trained execution*']
    for ax, setting, title, color in zip(axes, SETTINGS[:2], ['Later classifiers', 'Grammatical response'], ['#785481', '#286956']):
        for i, reference in enumerate(['initial_raw_readout', 'initial_tangent', 'initial_refined_common', 'task_adapted_tangent']):
            result = data[setting]['trained_reference'] if i==3 else data[setting]
            contrast = next(c for c in result['contrasts'] if c['reference']==reference and c['family']=='participation')
            x = contrast['reduction']
            lo, hi = contrast['interval']
            pair = np.array(result['summary'][reference]['participation']['by_seed'])-np.array(result['summary'][PROFILE]['participation']['by_seed'])
            ax.plot([lo, hi], [i, i], color=color, lw=1.4)
            ax.scatter(pair, i+np.linspace(-.11, .11, len(pair)), s=13, facecolors='white', edgecolors=color, linewidths=.65, zorder=3)
            ax.scatter([x], [i], marker='D', s=24, color=color, zorder=4)
        ax.axvline(0, color='#777777', lw=.7, linestyle=(0,(3,3)))
        ax.set_title(title, loc='left', pad=9)
        ax.set_yticks(range(4), labels if setting==SETTINGS[0] else ['']*4)
        ax.tick_params(axis='y', length=0)
        ax.set_ylim(3.55, -.55)
        ax.set_xlabel('Response-error reduction')
        ax.grid(axis='x', color='#dddddd', lw=.45)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=.265, right=.985, top=.84, bottom=.24, wspace=.25)
    fig.text(.265, .015, 'Reference error − source-profile error; positive values favor the profile.', fontsize=8)
    exports = []
    for suffix in ['pdf', 'svg', 'png']:
        dest = PAPER / 'figures' / ('source_profile_confirmation.'+suffix)
        fig.savefig(dest, dpi=220)
        exports.append(dict(path=str(dest.relative_to(ROOT)), sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    plt.close(fig)
    (ART / 'PROFILE_EXPORT.json').write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), exports=exports,
        scope='Frozen primary families; circles are target means, diamonds are overall means, lines are paired bootstrap intervals. Starred comparison uses targets2to5.'), indent=2)+'\n')
    provenance = dict(sources=[identity(source), identity(ART/'PROFILE_CONFIRMATION_FREEZE.json')],
        generator=identity(Path(__file__).resolve()),
        outputs=[identity(PAPER/'data/source_profile_confirmation.json'),
                 identity(PAPER/'data/source_profile_confirmation.csv'),
                 identity(PAPER/'tables/source_profile_confirmation.tex'),
                 identity(PAPER/'figures/source_profile_confirmation.pdf')])
    for path in [PAPER/'data/DATA_MANIFEST.json', PAPER/'figures/FIGURE_MANIFEST.json']:
        manifest = json.loads(path.read_text())
        manifest['source_profile_confirmation'] = provenance
        path.write_text(json.dumps(manifest, indent=2)+'\n')
    path = PAPER/'EVIDENCE_INDEX.json'
    index = json.loads(path.read_text())
    claim_id = 'source_profile_confirmation'
    index['claims'] = [c for c in index['claims'] if c['id']!=claim_id]
    index['claims'].append(dict(id=claim_id, paper='Section4.1 and AppendixE.5',
        result='One source response profile is reused across five unchanged target dictionaries. The frozen comparison measures its effect against equal-support Euclidean execution, active columns and source-direction readout, with a matched four-target trained-program comparison.',
        evidence=provenance['sources']+provenance['outputs'],
        statistics=data['statistics'], normalization_note=data['normalization_note']))
    if claim_id not in index['current_manuscript_claim_ids']:
        index['current_manuscript_claim_ids'].append(claim_id)
    index['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(index, indent=2)+'\n')
    print(json.dumps(dict(exported_rows=len(rows), figures=len(exports))))


if __name__ == '__main__':
    main()
