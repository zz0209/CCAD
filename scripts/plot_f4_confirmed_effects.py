"""Audited source-table figures, reusing the project's Pillow plotting approach.

No model/data-cache access. Confirmation and exploratory fits remain separate.
Every displayed query aggregate is replayed from saved endpoint observations.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ccad.artifacts import sha256, validate_run_directory


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def readl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def source_key(r):
    return tuple(r[k] for k in ('source_seed', 'source_atom', 'rank', 'condition', 'sequence'))


def verify_panel(spec, inputs):
    folder = ROOT / spec['summary_dir']
    cfg = json.loads((folder / 'config.resolved.json').read_text())
    if len(cfg['run_names']) != 1:
        raise ValueError('Panels must not silently pool multiple runs')
    causal = ROOT / 'runs' / cfg['run_names'][0]
    raw_path = causal / 'metrics.raw.jsonl'
    paths = [folder / n for n in ('config.resolved.json', 'summary.json', 'query.csv', 'target.csv', 'source_selection.jsonl')]
    paths += [raw_path, causal / 'status.json']
    if sha256(raw_path) != spec['raw_sha256'] or json.loads((causal / 'status.json').read_text())['status'] != 'PASS':
        raise ValueError('Raw identity/status mismatch')
    for p in paths:
        inputs.append(dict(path=str(p.resolve()), sha256=sha256(p), bytes=p.stat().st_size,
            source=spec['id'], license_or_access_boundary='local research; no publication authorization implied', role=p.name))
    source = readl(folder / 'source_selection.jsonl')
    choices = {}
    for r in source:
        rule = cfg['rule']
        selected = bool(r['supported'] and r['largest_atom_energy_share'] is not None and
            r['largest_atom_energy_share'] <= rule['maximum_largest_source_atom_energy_share'] and
            r['natural_source_hook_fraction'] is not None and
            r['natural_source_hook_fraction'] >= rule['minimum_natural_source_hook_fraction'])
        if selected != r['selected'] or source_key(r) in choices:
            raise ValueError('Source choice replay mismatch or duplicate')
        choices[source_key(r)] = selected
    raw = readl(raw_path)
    groups = defaultdict(list)
    for r in raw:
        choice = choices[source_key(r)]
        methods = [r['method']]
        policy = cfg.get('derived_policy')
        if policy and r['method'] == policy['selected_method' if choice else 'rejected_method']:
            methods.append(policy['name'])
        for method in methods:
            for endpoint in cfg['endpoints']:
                for subset in ('all', 'selected' if choice else 'rejected'):
                    k = (subset, r['condition'], r['source_seed'], r['source_atom'], r['target_seed'], method, endpoint)
                    groups[k].append(r['endpoints'][endpoint]['normalized_error'])
    query = defaultdict(list)
    for k, values in groups.items():
        query[k[:4] + k[5:]].append(median(values))
    with (folder / 'query.csv').open(newline='') as stream:
        saved = list(csv.DictReader(stream))
    if len(saved) != len(query):
        raise ValueError('Query denominator mismatch')
    rows = []
    for r in saved:
        r['source_seed'] = int(r['source_seed']); r['source_atom'] = int(r['source_atom'])
        k = tuple(r[f] for f in ('subset', 'condition', 'source_seed', 'source_atom', 'method', 'endpoint'))
        value = float(r['error']) if r['error'] else None
        if value != median(query[k]):
            raise ValueError(f'Query value replay mismatch: {k}')
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError('Log figure needs explicit new encoding for zero/nonpositive values')
        rows.append(dict(panel=spec['id'], evidence_tier=spec['tier'], **r, value=value))
    counts = dict(source_inputs=len(source), selected_inputs=sum(choices.values()),
        supported_inputs=sum(r['supported'] for r in source), query_cells_replayed=len(saved),
        source_queries=len({(r['source_seed'], r['source_atom']) for r in source}), raw_rows=len(raw))
    stats = []
    for subset, method, endpoint in sorted({(r['subset'], r['method'], r['endpoint']) for r in rows}):
        data = [r['value'] for r in rows if (r['subset'], r['method'], r['endpoint']) == (subset, method, endpoint)]
        stats.append(dict(panel=spec['id'], subset=subset, method=method, endpoint=endpoint,
            median=median(data), groups=len(data), valid_groups=sum(v is not None for v in data),
            below_zero=sum(v is not None and v < 1 for v in data)))
    return dict(spec=spec, rows=rows, counts=counts, stats=stats)


LABELS = {'source_adaptive_top16': 'Adaptive', 'target': 'Full', 'readout_top16': 'Top16',
    'raw': 'Raw', 'single_atom_dynamic': 'Atom\ndynamic', 'single_atom_level': 'Atom\nlevel',
    'uot_default': 'OT\nfixed', 'uot_discovery_tuned': 'OT\ntuned', 'wrong_query_matched_energy': 'Wrong\nquery',
    'conditional': 'Conditional\nrefit', 'contrast': 'Contrast\nrefit', 'bounded': 'Bounded\nrefit',
    'global_rows': 'Global\nrows'}


def render(run, cfg, panels, name, title, cells, footer, row_specific_limits=False):
    from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
    image = Image.new('RGB', (cfg['width_pixels'], cfg['height_pixels']), 'white')
    draw = ImageDraw.Draw(image); plotted = []
    def text(x, y, s, size=28, anchor='la', bold=False, fill='#202020'):
        font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf', size)
        draw.multiline_text((x, y), s, font=font, fill=fill, anchor=anchor, align='center', spacing=3)
    text(1080, 35, title, 43, 'ma', True)
    text(1080, 96, 'Circle: positive condition    Triangle: negative condition    Bar: group median', 29, 'ma')
    text(1080, 142, 'Source-normalized squared error; log10 scale; lower is better; dashed line: zero-effect reference = 1', 28, 'ma')
    all_values = [r['value'] for cell in cells for r in cell['rows'] if r['method'] in cell['methods'] and r['value'] is not None]
    for i, cell in enumerate(cells):
        col = i % 2; row = i // 2
        x = 126 + col * 1060; top = 305 + row * 660; width = 928; height = 434
        values = [r['value'] for c in (cells[row*2:row*2+2] if row_specific_limits else cells) for r in c['rows']
            if r['method'] in c['methods'] and r['value'] is not None]
        low = math.floor(math.log10(min(values))); high = math.ceil(math.log10(max(1, max(values))))
        if low == high: high += 1
        Y = lambda v: top + height * (high - math.log10(v)) / (high - low)
        text(x, top-108, cell['title'], 33, bold=True)
        text(x, top-62, cell['subtitle'], 25)
        for exponent in range(low, high+1):
            y = Y(10**exponent)
            draw.line((x, y, x+width, y), fill='#DDDDDD', width=2)
            text(x-16, y, f'1e{exponent}', 26, 'rm')
        for start in range(x, x+width, 24):
            draw.line((start, Y(1), min(start+13,x+width), Y(1)), fill='#202020', width=3)
        methods = cell['methods']
        for j, method in enumerate(methods):
            xx = x + (j+.5)*width/len(methods)
            data = sorted([r for r in cell['rows'] if r['method'] == method], key=lambda r:(r['source_seed'],r['source_atom'],r['condition']))
            color = '#0072B2' if method == cell.get('highlight') else '#666666'
            valid = [r for r in data if r['value'] is not None]
            if not data:
                text(xx, top+height-65, 'not\nrun', 25, 'ma', fill='#666666')
            for k, r in enumerate(data):
                if r['value'] is None: continue
                jitter = (k-(len(data)-1)/2)*min(4, 52/max(1,len(data)-1))
                px = xx+jitter; py = Y(r['value'])
                if r['condition'] == 'positive':
                    draw.ellipse((px-6,py-6,px+6,py+6), fill=color, outline=color, width=2)
                else:
                    draw.polygon([(px,py-8),(px-7,py+6),(px+7,py+6)], fill='white', outline=color, width=2)
                plotted.append(dict(figure=name, cell=i, **r, plotted_x=px, plotted_y=py, log10_low=low, log10_high=high))
            if valid:
                yy=Y(median([r['value'] for r in valid]));draw.line((xx-24,yy,xx+24,yy), fill='#0072B2' if method==cell.get('highlight') else '#202020', width=6)
            text(xx, top+height+20, LABELS[method], 26, 'ma', bold=method==cell.get('highlight'))
            if len(valid) != len(data): text(xx, top+height+85, f'{len(data)-len(valid)} missing', 21, 'ma')
        draw.line((x,top,x,top+height,x+width,top+height), fill='#202020', width=2)
    for j, line in enumerate(footer):text(1080, 1610+j*43, line, 27, 'ma')
    output=run/f'{name}.png';image.save(output,dpi=(cfg['dpi'],cfg['dpi']))
    with Image.open(output) as check:
        metadata=dict(pixels=list(check.size),mode=check.mode,dpi=list(check.info['dpi']),physical_width_mm=check.width/check.info['dpi'][0]*25.4)
    return plotted,dict(path=output.name,sha256=sha256(output),bytes=output.stat().st_size,
        plotted_points=len(plotted),metadata=metadata,pillow=pillow_version,row_specific_log_limits=row_specific_limits)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    started=time.perf_counter();write(run/'config.resolved.json',cfg)
    code=[]
    for path in (Path(__file__),ROOT/'src/ccad/artifacts.py'):
        rel=path.relative_to(ROOT).as_posix();dest=run/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
        code.append(dict(path=rel,sha256=sha256(path),bytes=path.stat().st_size,snapshot_path=dest.relative_to(run).as_posix()))
    code.sort(key=lambda r:r['path']);code_hash=hashlib.sha256(''.join(f"{r['path']}:{r['sha256']}\n" for r in code).encode()).hexdigest()
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=code_hash,snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.effect.figure.v1',run_id=cfg['run_id'],run_parent='R011-F4',
        purpose='Audited figures from confirmed and separately labeled development effects',milestone='M4-figure',
        evidence_level='visual synthesis; no new causal experiment',started_utc=datetime.now(timezone.utc).isoformat(),
        project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=code_hash,
        audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='unchanged original mean',
        threshold_source_split='saved frozen source rule',statistics_unit='query-condition aggregate; dependent seed/documents',
        device='CPU Pillow only',seeds=[1,2,3,4,5],resource_lease='none',resource_lease_reason=cfg['budget'],source_snapshot_required=True))
    write(run/'status.json',dict(status='RUNNING'))
    inputs=[];result={};error=None
    try:
        panels={s['id']:verify_panel(s,inputs) for s in cfg['panels']}
        cells=[];methods=cfg.get('methods',list(LABELS)[:9])
        for key in ('original','expanded'):
            p=panels[key];n=p['counts']
            for ep,title in [('centered_logits','Logits'),('next_state','Next state')]:
                cells.append(dict(title=f"{p['spec']['label']} | {title}",subtitle=f"16 groups; {n['supported_inputs']}/64 source inputs supported; {n['selected_inputs']}/64 selected",
                    rows=[r for r in p['rows'] if r['subset']=='all' and r['endpoint']==ep],methods=methods,highlight='source_adaptive_top16'))
        foot=['Two fixed-query new-document confirmations; both panels use the same five SAE seeds and hook.',
            'Points are document-then-target medians, not independent replicates; bars are descriptive, not confidence intervals.',
            'Original panel: 3 unsupported source inputs (12 target rows per method missing); expanded panel: none.']
        plotted,exports=render(run,cfg,panels,'confirmed_effects',cfg.get('confirmation_title','Frozen source-adaptive correspondence: confirmed effects'),cells,cfg.get('confirmation_footer',foot))
        allpoints=plotted;outputs=[exports];cells=[]
        for key in ('original','expanded'):
            p=panels[key];n=p['counts']
            for subset,title in [('selected','Source-selected full range'),('rejected','Fallback range')]:
                rows=[r for r in p['rows'] if r['subset']==subset and r['endpoint']=='next_state']
                groups=len({(r['source_seed'],r['source_atom'],r['condition']) for r in rows})
                ni=n['selected_inputs'] if subset=='selected' else 64-n['selected_inputs']
                cells.append(dict(title=f"{p['spec']['label']} | {title}",subtitle=f"Next-state endpoint; {ni}/64 input pairs; {groups} query-condition groups",rows=rows,
                    methods=cfg.get('scope_methods',['target','readout_top16','raw','single_atom_dynamic','uot_discovery_tuned']),highlight='target' if subset=='selected' else 'readout_top16'))
        plotted,exports=render(run,cfg,panels,'source_scope','Separate the full-relation range from the sparse fallback',cells,
            ['Same frozen source rule; each method is compared on the same within-panel subset.',
             'A query-condition group may occur in both subsets. No independent-seed confidence intervals.',
             'All group points shown. Logit counterpart and complete denominators are available in the source tables.'])
        allpoints+=plotted;outputs.append(exports);cells=[]
        if not cfg.get('skip_refit_development',False):
            for key in ('original','expanded'):
                for ep,title in [('centered_logits','Logits'),('next_state','Next state')]:
                    rows=[r for r in panels[key]['rows'] if r['subset']=='all' and r['endpoint']==ep and r['method'] in ('target','readout_top16')]
                    for extra,method in [(key+'_refit','conditional')]+([('contrast','contrast'),('bounded','bounded')] if key=='original' else []):
                        rows += [dict(r,method=method) for r in panels[extra]['rows'] if r['subset']=='all' and r['endpoint']==ep]
                    cells.append(dict(title=f"{panels[key]['spec']['label']} | {title}",subtitle='Development comparison; same 16 groups; log limits differ by row',rows=rows,
                        methods=['target','readout_top16','conditional','contrast','bounded'],highlight='conditional'))
            plotted,exports=render(run,cfg,panels,'refit_development','Development boundary: truncation is not optimal sparse fitting',cells,
                ['Refits were developed after document exposure; these are not new independent confirmations.',
                 'Original panel retains severe unbounded failures. Contrast/bounded fits were not run on the expanded panel.',
                 'Each row has its own explicitly labeled log10 limits. Points and medians are descriptive, not confidence intervals.'],True)
            allpoints+=plotted;outputs.append(exports)
        rows=[r for p in panels.values() for r in p['rows']]
        (run/'metrics.raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        for name,data in [('query_source_data',rows),('plotted_points',allpoints),('method_summary',[r for p in panels.values() for r in p['stats']])]:
            with (run/f'{name}.csv').open('w',newline='') as stream:
                w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        result=dict(status='PASS',panels={k:p['counts'] for k,p in panels.items()},outputs=outputs,
            replayed_query_cells=len(rows),plotted_points=len(allpoints),new_model_forwards=0,
            transformation='Exact raw endpoint -> median input within query/target/condition -> median target. All groups retained; no smoothing, winsorization, CI or pooled independent-seed inference.',
            raw_metric_role='Derived query aggregates; upstream raw observations remain immutable',scope=cfg['destination'])
    except Exception:
        error=traceback.format_exc();result=dict(status='FAIL',error=error)
    write(run/'inputs.json',dict(inputs=inputs));(run/'stderr.log').write_text(error or '')
    if not (run/'metrics.raw.jsonl').exists():(run/'metrics.raw.jsonl').write_text('')
    from PIL import __version__ as pillow_version
    write(run/'environment.json',dict(python=sys.executable,python_version=platform.python_version(),pillow=pillow_version,
        platform=platform.platform(),ML_runtime='not loaded or modified',matplotlib='not used',fonts=['C:/Windows/Fonts/arial.ttf','C:/Windows/Fonts/arialbd.ttf']))
    result.update(wall_seconds=time.perf_counter()-started,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),
        generator_script_path=Path(__file__).relative_to(ROOT).as_posix(),generator_script_sha256=sha256(Path(__file__)))
    write(run/'metrics.summary.json',result);write(run/'status.json',dict(status=result['status'],error=error));write(run/'stdout.log',result)
    validation=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=validation.ok,errors=list(validation.errors)))
    print(json.dumps(result,indent=2));return 0 if result['status']=='PASS' and validation.ok else 1


if __name__=='__main__':raise SystemExit(main())
