"""Draw the opening example and reuse benefit from the frozen R57 export."""
from pathlib import Path
import json
import hashlib
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'


def main():
    path = PAPER / 'data/binding_member_selection.json'
    d = json.loads(path.read_text())
    example = d['example']
    assert example['row']['names'] == ['Henry', 'Emma']
    assert example['decoded_outputs']['source_path'] == {
        'entity': ['Santiago', 'Warsaw'], 'both': ['Warsaw', 'Santiago']}
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8,
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.linewidth': .5})
    fig = plt.figure(figsize=(3.375, 2.90))
    fig.text(.02, .98, '(a) Reuse an explanation\'s intervention requests', va='top', fontsize=8.5)
    fig.text(.02, .905, 'Henry: Poland (Warsaw)     Emma: Chile (Santiago)', fontsize=8)
    fig.text(.02, .82, 'Request', weight='bold')
    fig.text(.60, .82, 'Henry', ha='center', weight='bold')
    fig.text(.86, .82, 'Emma', ha='center', weight='bold')
    fig.add_artist(plt.Line2D([.02, .98], [.795, .795], transform=fig.transFigure, color='.4', lw=.5))
    for y, request, key in [(.744, 'Swap people', 'entity'), (.67, 'Also swap attributes', 'both')]:
        fig.text(.02, y, request)
        for x, answer in zip([.60, .86], example['decoded_outputs']['source_path'][key]):
            fig.text(x, y, answer, ha='center', color='#216b57')
    fig.text(.02, .607, 'New target SAE; 16 members selected using saved responses.', fontsize=7.5)
    fig.text(.02, .536, '(b) Complete rule accuracy across four target dictionaries', fontsize=8.5)
    ax = fig.add_axes([.40, .11, .40, .31])
    methods = [('geometry', 'Magnitude', '#777777', 'o'),
               ('source_cached', 'Cached gradient', '#777777', 's'),
               ('source_path', 'Saved source path', '#216b57', 'o'),
               ('raw_path', 'Saved raw-role path', '#216b57', '^'),
               ('target_path', 'New target path', '#78517b', 'D')]
    rows = []
    for i, (name, label, color, marker) in enumerate(methods):
        v = d['results']['all_forms']['methods'][name + '/k16']['complete']
        point, (lo, hi) = v['percent'], v['ci95']
        ax.errorbar(point, i, xerr=[[point-lo], [hi-point]], fmt=marker, color=color,
                    markersize=3.4, elinewidth=.65, capsize=1.8)
        calls = 256 if name == 'target_path' else 0
        ax.text(1.15, i, str(calls), transform=ax.get_yaxis_transform(), va='center', ha='center', fontsize=8)
        rows.append(dict(method=name, complete_accuracy_percent=point, ci_lower_percent=lo,
                         ci_upper_percent=hi, new_target_gradient_minibatches=calls))
    ax.set_yticks(range(len(methods)), [x[1] for x in methods], fontsize=7.6)
    ax.set_ylim(4.5, -.5)
    ax.set_xlim(60, 100)
    ax.set_xticks([60, 80, 100])
    ax.tick_params(axis='y', length=0, pad=4)
    ax.tick_params(axis='x', length=2, pad=2, labelsize=7)
    ax.set_xlabel('Accuracy (%)', fontsize=8, labelpad=1)
    ax.text(1.15, 1.10, 'New target\ngradients', transform=ax.transAxes, ha='center', va='bottom', fontsize=7.4)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    outputs = []
    for ext in ['pdf', 'svg', 'png']:
        out = PAPER / 'figures' / ('reuse_opening.' + ext)
        fig.savefig(out, dpi=250)
        outputs.append(dict(path=out.relative_to(ROOT).as_posix(), bytes=out.stat().st_size,
                            sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
    plt.close(fig)
    output = PAPER / 'data/reuse_opening.csv'
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (PAPER / 'data/REUSE_OPENING_MANIFEST.json').write_text(json.dumps(dict(
        source=dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        generator=dict(path=Path(__file__).relative_to(ROOT).as_posix(), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        outputs=outputs, example=example, example_selection=d['example_selection'],
        scope='Existing R57 data only. Example was selected descriptively after outcomes; all-panel estimates use every frozen world. '
              'New-gradient cost applies after a saved bank exists; initial source bank cost320 minibatches is reported in the main text.'
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
