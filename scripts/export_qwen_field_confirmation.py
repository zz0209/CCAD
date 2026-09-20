import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.aris/plot_runtime_v1'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def identity(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    art = ROOT/'artifacts/final_science_20260920_round03'
    paper = ROOT/'paper'
    analysis = art/'CONFIRMATION_ANALYSIS_V2.json'
    data = json.loads(analysis.read_text())
    methods = [('Member relation', 'relation'), ('Source-direction readout', 'readout'),
               ('Euclidean inference', 'euclidean'), ('Source-profile inference', 'profile')]
    shutil.copyfile(analysis, paper/'data/qwen_field_confirmation.json')
    with (paper/'data/qwen_field_confirmation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data['cells'][0]))
        writer.writeheader()
        writer.writerows(data['cells'])
    table = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'& \multicolumn{2}{c}{Two terms} & \multicolumn{2}{c}{Three terms} \\',
             r'Execution & Full & Parts & Full & Parts \\', r'\midrule']
    values = {}
    for label, method in methods:
        row = []
        for arity in [2, 3]:
            for family in ['full', 'parts']:
                key = 'exact_agreement' if family == 'full' else 'balanced_agreement'
                cells = [c[key] for c in data['cells'] if c['arity'] == arity and c['family'] == family and c['method'] == method]
                assert len(cells) == 2 and all(v is not None for v in cells)
                value = 100*float(np.mean(cells))
                values[method, arity, family] = value
                row.append(f'{value:.2f}')
        table.append(label+' & '+' & '.join(row)+r' \\')
    table.extend([r'\bottomrule', r'\end{tabular}'])
    (paper/'tables/qwen_field_confirmation.tex').write_text('\n'.join(table)+'\n')
    complete = [r'\begin{tabular}{lrrrrr}', r'\toprule',
                r'& Full agreement & Part changed & Part unchanged & Balanced parts & Full hybrid \\',
                r'\midrule']
    for arity in [2, 3]:
        for operation in ['unit', 'tens']:
            label = 'units' if operation == 'unit' else 'tens'
            complete.append(r'\multicolumn{6}{l}{'+f'{arity}-term inputs, {label} replacement'+r'} \\')
            for label, method in methods:
                part = next(c for c in data['cells'] if c['arity'] == arity and
                            c['operation'] == operation and c['family'] == 'parts' and c['method'] == method)
                full = next(c for c in data['cells'] if c['arity'] == arity and
                            c['operation'] == operation and c['family'] == 'full' and c['method'] == method)
                numbers = [full['exact_agreement'], part['changed_agreement'], part['unchanged_agreement'],
                           part['balanced_agreement'], full['hybrid_success']]
                complete.append(label+' & '+' & '.join(f'{100*v:.2f}' for v in numbers)+r' \\')
            complete.append(r'\addlinespace')
    complete.extend([r'\bottomrule', r'\end{tabular}'])
    (paper/'tables/qwen_field_complete.tex').write_text('\n'.join(complete)+'\n')
    macros = []
    for label, key in [('Profile', 'profile'), ('Euclidean', 'euclidean'), ('Relation', 'relation'), ('Readout', 'readout')]:
        macros.append('\\newcommand{\\QwenField'+label+'}{'+f'{100*data["primary"][key]:.2f}'+'}')
        for arity, name in [(2, 'Two'), (3, 'Three')]:
            macros.append('\\newcommand{\\QwenField'+label+name+'}{'+f'{values[key, arity, "parts"]:.2f}'+'}')
    for contrast in data['contrasts']:
        label = dict(euclidean='Euclidean', relation='Relation', readout='Readout')[contrast['comparator']]
        for suffix, value in [('Gain', contrast['difference']), ('Lower', contrast['interval'][0]), ('Upper', contrast['interval'][1])]:
            macros.append('\\newcommand{\\QwenField'+label+suffix+'}{'+f'{100*value:.2f}'+'}')
    for arity, name in [('2', 'Two'), ('3', 'Three')]:
        macros.append('\\newcommand{\\QwenSourceHybrid'+name+'}{'+f'{100*data["source_full_hybrid"][arity]:.2f}'+'}')
    (paper/'tables/qwen_field_values.tex').write_text('\n'.join(macros)+'\n')
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family':'Times New Roman', 'font.size':9, 'mathtext.fontset':'stix',
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.spines.left':False,
        'axes.linewidth':.6, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75), gridspec_kw={'width_ratios':[1.15, 1]})
    colors = {'profile':'#286956', 'euclidean':'#343434', 'relation':'#967143', 'readout':'#795a89'}
    markers = {'profile':'D', 'euclidean':'o', 'relation':'s', 'readout':'^'}
    shifts = {'relation':-.15, 'readout':-.05, 'euclidean':.05, 'profile':.15}
    for label, method in methods:
        values_by_seed = [100*r['values'][method] for r in data['directions']]
        axes[0].scatter(values_by_seed, np.arange(5)+shifts[method], label=label,
            marker=markers[method], c=colors[method], s=22, linewidths=.5)
    axes[0].set_yticks(range(5), [f'{r["source"]} → {r["target"]}' for r in data['directions']])
    axes[0].set_ylim(4.5, -.5)
    axes[0].set_xlabel('Balanced part agreement (%)')
    axes[0].set_ylabel('Source → target seed')
    axes[0].grid(axis='x', color='#dddddd', lw=.5)
    axes[0].set_axisbelow(True)
    refs = ['euclidean', 'relation', 'readout']
    for i, ref in enumerate(refs):
        contrast = next(c for c in data['contrasts'] if c['comparator'] == ref)
        lo, hi = 100*np.asarray(contrast['interval'])
        axes[1].plot([lo, hi], [i, i], color=colors[ref], lw=1.4)
        axes[1].scatter([100*contrast['difference']], [i], color=colors[ref], marker=markers[ref], s=25)
    axes[1].axvline(0, color='#777777', lw=.7, linestyle=(0, (3, 3)))
    axes[1].set_yticks(range(3), ['Euclidean', 'Member relation', 'Readout'])
    axes[1].set_ylim(2.5, -.5)
    axes[1].set_xlabel('Source-profile gain (points)')
    axes[1].tick_params(axis='y', length=0)
    axes[1].grid(axis='x', color='#dddddd', lw=.5)
    axes[1].set_axisbelow(True)
    axes[0].set_title('Fixed five-dictionary pool', loc='left', fontsize=10)
    axes[1].set_title('Paired uncertainty', loc='left', fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -.01), ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.105, right=.985, bottom=.31, top=.86, wspace=.63)
    exports = []
    for extension in ['pdf', 'svg', 'png']:
        path = paper/'figures'/f'qwen_field_confirmation.{extension}'
        fig.savefig(path, dpi=220)
        exports.append(identity(path))
    plt.close(fig)
    provenance = dict(sources=[identity(analysis), identity(art/'CONFIRMATION_FREEZE.json'),
        identity(art/'READOUT_CORRECTION_FREEZE.json')],
        generator=identity(Path(__file__)), outputs=exports+[identity(paper/'data/qwen_field_confirmation.csv'),
            identity(paper/'tables/qwen_field_confirmation.tex'), identity(paper/'tables/qwen_field_complete.tex'),
            identity(paper/'tables/qwen_field_values.tex')])
    for name in ['data/DATA_MANIFEST.json', 'figures/FIGURE_MANIFEST.json']:
        path = paper/name
        manifest = json.loads(path.read_text())
        manifest['qwen_field_confirmation'] = provenance
        path.write_text(json.dumps(manifest, indent=2)+'\n')
    index_path = paper/'EVIDENCE_INDEX.json'
    index = json.loads(index_path.read_text())
    claim_id = 'qwen_field_confirmation'
    index['claims'] = [c for c in index['claims'] if c['id'] != claim_id]
    index['claims'].append(dict(id=claim_id, paper='Section4.2 and AppendixC',
        result='The common source-field inference rule is tested during complete Qwen generation on new member requests, known two-term questions in new contexts, and previously unused three-term questions. Its frozen primary comparison shares exact source amplitudes and target member allowance with Euclidean inference.',
        evidence=provenance['sources']+provenance['outputs'], statistics=data['statistics'],
        source_information='Inference receives exact source amplitudes. The inherited source-direction readout estimates them from a fixed target-code bank.',
        primary=data['primary'], contrasts=data['contrasts']))
    if claim_id not in index['current_manuscript_claim_ids']:
        index['current_manuscript_claim_ids'].append(claim_id)
    index['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    index_path.write_text(json.dumps(index, indent=2)+'\n')
    (art/'EXPORT.json').write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(), **provenance), indent=2)+'\n')
    print(json.dumps(dict(primary=data['primary'], figure=exports[0])))


if __name__ == '__main__':
    main()
