"""Natural-stream reconstruction and task-group function along one training stream."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.aris/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out = ROOT / 'artifacts/seven_round_rebuild_20260906/r4_l15'
    paths = [out / 'TRAINING_SUMMARY.json', out / 'FUNCTIONAL_LEARNING_SUMMARY.json']
    quality, function = [json.loads(p.read_text()) for p in paths]
    steps = [256, 1024, 4096]
    x = np.array(steps) * 1024 / 1e6
    values = []
    style = {'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 8,
             'axes.linewidth': .5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
        fig.subplots_adjust(left=.09, right=.98, bottom=.25, top=.84, wspace=.29)
        for ax, series in zip(axes, [
            [('fve', 'FVE', '#0072B2', 'o'), ('ce_recovered', 'CE recovery', '#D55E00', 's')],
            [('number', 'Number, 16 members', '#0072B2', 'o'),
             ('time', 'Time, 32 members', '#009E73', '^'),
             ('joint', 'Joint operation', '#D55E00', 's')]]):
            for key, label, color, marker in series:
                trajectories = []
                for seed in range(1, 6):
                    if key in ['fve', 'ce_recovered']:
                        y = [next(r['quality'][key] for r in quality['rows'] if r['seed'] == seed and r['step'] == step) for step in steps]
                    else:
                        y = [next(r['accuracy'] for r in function['rows'] if r['partition'] == f'seed{seed}' and r['step'] == step and r['factor'] == key) for step in steps]
                    trajectories.append(y)
                    ax.plot(x, np.array(y) * 100, color=color, alpha=.22, lw=.55)
                    values.append(dict(series=key, seed=seed, steps=steps, values=y))
                ax.plot(x, np.mean(trajectories, axis=0) * 100, color=color, marker=marker, ms=3.4, lw=1.15, label=label)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_ylim(0, 100)
            ax.set_xticks(x, ['0.262', '1.049', '4.194'])
            ax.set_xlabel('Natural training tokens (millions)')
            ax.legend(frameon=False, loc='lower right', fontsize=6.6)
        axes[0].set_ylabel('Fixed-validation recovery (%)')
        axes[1].set_ylabel('Expected-answer accuracy (%)')
        axes[0].set_title('a   Whole-stream reconstruction', loc='left', pad=10)
        axes[1].set_title('b   Compact source operations', loc='left', pad=10)
        files = []
        for ext in ['png', 'pdf', 'svg']:
            p = out / ('figure_checkpoint_function.' + ext)
            fig.savefig(p, dpi=300, facecolor='white')
            files.append(dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        plt.close(fig)
    (out / 'figure_checkpoint_function_manifest.json').write_text(json.dumps(dict(
        sources=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        files=files, values=values,
        description='Same ordered natural training stream, five seeds and three dependent checkpoints. Thin lines are individual seeds; marked lines means, no confidence interval. Right panel uses all384 temporal development prompts with the same functional selection rule at each checkpoint; feature IDs are not fixed. Material learning is separate from frozen confirmation.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
