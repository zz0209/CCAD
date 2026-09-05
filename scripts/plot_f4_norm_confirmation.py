"""Provisional internal figure: all four directions and both collateral endpoints."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, 'C:/Users/zz/.codex/skills/scientific-visualization/assets')
from color_palettes import OKABE_ITO_ON_WHITE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    runs = [ROOT/'runs'/name for name in (
        'F4_task_norm_matched_causal_v1_20260905',
        'F4_task_paired_reserved_confirmation_v1_20260905')]
    out = runs[1]; output = out/'norm_development_confirmation.png'
    if output.exists():
        raise FileExistsError(output)
    data = [json.loads((r/'norm_comparison.json').read_text()) for r in runs]
    colors = OKABE_ITO_ON_WHITE[:2]
    fontpath = 'C:/Windows/Fonts/arial.ttf'
    font = lambda n: ImageFont.truetype(fontpath, n)
    im = Image.new('RGB', (1800, 1290), 'white'); d = ImageDraw.Draw(im)
    d.text((40, 24), 'Matched-norm FCC: development gain did not broadly confirm', font=font(32), fill='black')
    d.text((40, 73), 'Raw minus code-FCC: positive means lower FCC error / smaller FCC collateral change.', font=font(24), fill='black')

    def marker(x, y, k, radius=7):
        box = (x-radius, y-radius, x+radius, y+radius)
        if k == 0: d.ellipse(box, fill=colors[k])
        else: d.rectangle(box, fill=colors[k])

    for k, label in enumerate(('Development: 64 inputs / 16 templates', 'Frozen confirmation: 64 new inputs / 16 templates')):
        x = 55 + 740*k; marker(x, 127, k)
        d.text((x+18, 111), label, font=font(22), fill='black')
    error_limits = [v for obj in data for row in obj['comparisons'] for v in row['bootstrap_95_pointwise']]
    lo, hi = min(error_limits), max(error_limits)
    lo -= .07*(hi-lo); hi += .07*(hi-lo)
    panels = [('A  Subject-effect prediction error', 'primary', (lo, hi)),
              ('B  Past-effect prediction error', 'past', (lo, hi)),
              ('C  Attractor: absolute tense change', 'mean_abs_tense', (-.085, .04)),
              ('D  Attractor: absolute subject change', 'mean_abs_primary', (-.085, .04))]
    table = []
    for p, (title, endpoint, limits) in enumerate(panels):
        ox = 40 + (p % 2)*880; oy = 180 + (p // 2)*475
        left, right, top, bottom = ox+110, ox+805, oy+83, oy+365
        d.text((ox, oy), title, font=font(26), fill='black')
        subtitle = 'Normalized squared-error difference; 95% template interval' if p < 2 else 'Mean absolute log-probability margin change; no interval'
        d.text((ox, oy+38), subtitle, font=font(19), fill='black')
        ymin, ymax = limits
        yp = lambda v: bottom-(v-ymin)/(ymax-ymin)*(bottom-top)
        for j in range(6):
            value = ymin+(ymax-ymin)*j/5
            y = yp(value); d.line((left, y, right, y), fill='#dddddd', width=1)
            d.text((ox, y-10), f'{value:.3f}', font=font(19), fill='black')
        d.line((left, yp(0), right, yp(0)), fill='#555555', width=2)
        d.line((left, top, left, bottom), fill='black', width=1)
        for i, seed in enumerate((1, 3, 4, 5)):
            xcenter = left+(i+.5)*(right-left)/4
            d.text((xcenter-34, bottom+17), f'Target {seed}', font=font(20), fill='black')
            for k, obj in enumerate(data):
                x = xcenter+(-15 if k == 0 else 15)
                if p < 2:
                    row = next(r for r in obj['comparisons'] if r['target_seed']==seed and r['endpoint']==endpoint)
                    value = row['raw_minus_fcc_error']; lower, upper = row['bootstrap_95_pointwise']
                    d.line((x, yp(lower), x, yp(upper)), fill=colors[k], width=3)
                    for end in (lower, upper): d.line((x-7, yp(end), x+7, yp(end)), fill=colors[k], width=2)
                else:
                    rows = {r['method']:r for r in obj['rows'] if r['target_seed']==seed and r['axis']=='attractor'}
                    value = rows['matchedraw'][endpoint]-rows['codeFCC'][endpoint]
                    lower, upper = None, None
                assert ymin <= value <= ymax
                marker(x, yp(value), k)
                table.append(dict(panel=p+1, split=('development','confirmation')[k], target_seed=seed,
                                  endpoint=endpoint, raw_minus_fcc=value, lower=lower, upper=upper))
    footer = [
        'A/B: error = sum(candidate effect - source effect)^2 / sum(source effect)^2; matched actual hook norms.',
        'Intervals: 5,000 whole-template bootstrap draws, four number conditions kept together; pointwise, not simultaneous.',
        'Four directions share source seed 2; they are not four independent replications. No inputs or directions excluded.',
        'Confirmation changes nouns/prepositions, not syntax family. C/D use different attractor operations; both are retained.',
        'Source-aligned transport, not target-native deletion or semantic decomposition. Original unmatched raw remains in full tables.'
    ]
    for i, line in enumerate(footer): d.text((40, 1140+i*27), line, font=font(20), fill='black')
    im.save(output, dpi=(150,150))
    with (out/'NORM_FIGURE_DATA.csv').open('w', newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    provenance = dict(audience='provisional internal research review', publisher='unspecified; submission requirements not claimed',
        size_px=list(im.size), dpi=150, mode='RGB', font=fontpath, pillow_version=pillow_version,
        palette='scientific-visualization OKABE_ITO_ON_WHITE first two; circles and squares',
        inputs=[dict(path=str(r/'norm_comparison.json'),sha256=digest(r/'norm_comparison.json')) for r in runs],
        generator_sha256=digest(Path(__file__)), figure_sha256=digest(output),
        transformations='No new statistical analysis; plot all stored contrasts and pointwise bootstrap intervals. Collateral contrasts are matchedraw minus FCC means.',
        missing_data='none; no filtering', uncertainty='A/B stored 95% template percentile bootstrap; C/D descriptive means without intervals',
        alt_text='All four development primary contrasts favor code FCC. Only target 5 confirmation primary contrast favors FCC, with interval crossing zero. FCC shows smaller attractor tense changes but larger attractor subject changes in both samples. Four directions share a source seed.')
    (out/'norm_figure_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
