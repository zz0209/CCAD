"""Re-layout existing region estimates for the manuscript; no new inference."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
LABELS = [('1', 'Verb only'), ('2', 'Number only'), ('4', 'Gender only'),
          ('3', 'Verb + number'), ('5', 'Verb + gender'),
          ('6', 'Number + gender'), ('7', 'All three')]


def main():
    source = PAPER / 'data/final_value_r29.json'
    data = json.loads(source.read_text())
    cfg = json.loads((ROOT / json.loads((PAPER / 'reform_runs.json').read_text())[
        'independent_consensus_run'] / 'config.resolved.json').read_text())
    tasks = cfg['source_tasks']
    rows = data['structure_aggregates']
    selected = [r for r in rows if r['kind'] == 'region']
    maximum = max(abs(v) for r in selected for v in r['task_margin_decrements'].values())
    font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8,
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.linewidth': .5})
    cmap = LinearSegmentedColormap.from_list('signed_effect', ['#78517b', '#fbfaf7', '#216b57'])
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
    fig, axes = plt.subplots(1, 4, figsize=(7.05, 2.65))
    fig.subplots_adjust(left=.17, right=.99, top=.82, bottom=.25, wspace=.18)
    records = []
    for col, (obj, family) in enumerate([
            ('topk', 'shared_path_full'), ('topk', 'cached64_full'),
            ('matryoshka', 'shared_path_full'), ('matryoshka', 'cached64_full')]):
        ax = axes[col]
        rr = [next(r for r in selected if r['objective'] == obj and
                   r['family'] == family and r['label'] == label) for label, _ in LABELS]
        matrix = np.array([[r['task_margin_decrements'][t] for t in tasks] for r in rr])
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
        for i, r in enumerate(rr):
            for j, value in enumerate(matrix[i]):
                ax.text(j, i, '\u2014' if r['nonempty_targets'] == 0 else f'{value:.2f}',
                        ha='center', va='center', fontsize=7.4,
                        color='white' if value > .65 * maximum else '#222222')
                records.append(dict(sae=obj, relation=family, region=LABELS[i][0],
                                    task=tasks[j], margin_decrease=float(value),
                                    nonempty_targets=r['nonempty_targets']))
        ax.set_xticks(range(3), ['Verb', 'Number', 'Gender'], fontsize=7.5)
        ax.set_yticks(range(7), [label for _, label in LABELS] if col == 0 else [])
        ax.tick_params(length=0, pad=3)
        ax.set_title('Source response' if col % 2 == 0 else 'Cached gradient', fontsize=8, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.366, .96, 'TopK', ha='center', fontsize=9)
    fig.text(.794, .96, 'Matryoshka', ha='center', fontsize=9)
    cax = fig.add_axes([.36, .095, .40, .026])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-2, -1, 0, 1, 2])
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    cb.set_label('Margin decrease (nats)', fontsize=8, labelpad=1)
    outputs = []
    for ext in ['pdf', 'svg', 'png']:
        path = PAPER / 'figures' / f'reuse_regions.{ext}'
        fig.savefig(path, dpi=220)
        outputs.append(dict(path=path.relative_to(ROOT).as_posix(),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    plt.close(fig)
    path = PAPER / 'data/reuse_regions.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (PAPER / 'data/REUSE_FIGURE_MANIFEST.json').write_text(json.dumps(dict(
        source=dict(path=source.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
        outputs=outputs, rows=len(records),
        scope='Layout of all existing region means for both relations and both mechanisms; no new model outputs or statistics.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
