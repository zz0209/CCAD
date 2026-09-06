"""Descriptive material screen; lexical blocks, not patch directions, are repeats."""
import csv, hashlib, json, statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/SEVEN_R1_functional_material_v2_20260906'
OUT=ROOT/'artifacts/seven_round_rebuild_20260906'

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    raw=[json.loads(line) for line in (RUN/'metrics.raw.jsonl').read_text().splitlines()]
    design=json.loads((RUN/'crossed_inputs.json').read_text())
    inputs={r['id']:r for r in design['rows']}
    public=defaultdict(list);cross=defaultdict(list);patches=defaultdict(list)
    for r in raw:
        if r['kind']=='public_baseline':public[r['model'],r['task']].append(r)
        elif r['kind']=='crossed_baseline':cross[r['model'],r['template']].append(r)
        else:patches[r['model'],r['layer'],r['site'],r['factor'],r['template']].append(r)
    public_table=[]
    for (model,task),rows in sorted(public.items()):
        unique_sides={(r['text'],r['expected'],r['other']) for r in rows}
        public_table.append(dict(model=model,task=task,n_recorded_sides=len(rows),n_unique_sides=len(unique_sides),
            n_released_pairs=len(set(r['public_pair'] for r in rows)),correct=sum(r['correct'] for r in rows),
            accuracy=statistics.mean(r['correct'] for r in rows),full_vocab_accuracy=statistics.mean(r['full_correct'] for r in rows)))
    cross_table=[]
    for (model,template),rows in sorted(cross.items()):
        cross_table.append(dict(model=model,template=template,n=len(rows),lexical_blocks=len(set(r['block'] for r in rows)),
            correct=sum(r['correct'] for r in rows),accuracy=statistics.mean(r['correct'] for r in rows),
            min_abs_margin=min(abs(r['plural_margin']) for r in rows)))
    patch_table=[];blocks=[]
    for key,rows in sorted(patches.items()):
        fields=dict(zip(['model','layer','site','factor','template'],key))
        patch_table.append(dict(**fields,n_directed_patches=len(rows),n_undirected_pairs=len({tuple(sorted((r['base'],r['donor']))) for r in rows}),
            n_lexical_blocks=len({r['block'] for r in rows}),correct=sum(r['donor_label_correct'] for r in rows),
            donor_accuracy=statistics.mean(r['donor_label_correct'] for r in rows),base_accuracy=statistics.mean(r['base_label_correct'] for r in rows),
            mean_abs_margin_change=statistics.mean(abs(r['plural_margin_change']) for r in rows),
            median_oriented_margin_change=statistics.median(r['donor_oriented_margin_change'] for r in rows),
            mean_kl_to_base=statistics.mean(r['kl_to_base'] for r in rows),
            mean_kl_to_full_donor=statistics.mean(r['kl_to_full_donor'] for r in rows),
            mean_patch_norm=statistics.mean(r['patched_vector_norm'] for r in rows)))
        for b in sorted({r['block'] for r in rows}):
            rr=[r for r in rows if r['block']==b]
            blocks.append(dict(**fields,block=b,lemma=inputs[rr[0]['base']]['subject_lemma'],n=len(rr),
                correct=sum(r['donor_label_correct'] for r in rr),
                mean_oriented_margin_change=statistics.mean(r['donor_oriented_margin_change'] for r in rr),
                mean_abs_margin_change=statistics.mean(abs(r['plural_margin_change']) for r in rr)))
    overview=[]
    for model in ['160m','1b']:
        pp=[r for r in raw if r['kind']=='public_baseline' and r['model']==model]
        cc=[r for r in raw if r['kind']=='crossed_baseline' and r['model']==model]
        overview.append(dict(model=model,public_correct=sum(r['correct'] for r in pp),public_n=len(pp),public_accuracy=statistics.mean(r['correct'] for r in pp),
            crossed_correct=sum(r['correct'] for r in cc),crossed_n=len(cc),crossed_accuracy=statistics.mean(r['correct'] for r in cc)))
    # First lexical block and first syntax template; no model-output selection.
    example_inputs=[r for r in design['rows'] if r['block']==0 and r['template']=='pp']
    example_ids={r['id'] for r in example_inputs}
    example_baselines=[r for r in raw if r['kind']=='crossed_baseline' and r['model']=='1b' and r['row_id'] in example_ids]
    example_patches=[r for r in raw if r['kind']=='crossed_patch' and r['model']=='1b' and r['base']==0 and r['layer'] in [3,7,11]]
    summary=dict(written_utc=datetime.now(timezone.utc).isoformat(),run=str(RUN),scope='Development material screen. No SAE or FCC result.',
        raw_sha256=hashlib.sha256((RUN/'metrics.raw.jsonl').read_bytes()).hexdigest(),overview=overview,public_tasks=public_table,
        crossed_baseline=cross_table,patches=patch_table,blocks=blocks,
        example=dict(selection='First configured lexical block and PP template, base row 0; selected without outcomes.',inputs=example_inputs,baseline=example_baselines,patches=example_patches),
        dependence='29 released tasks,100 dev rows each,with reciprocal prompt reuse; crossed16lexical blocks x3syntax x2subject x2distractor. Directions and sites sharing a token are not independent repeats.',
        aggregation='Accuracy on expected versus competing next-token label; raw counts and absolute log-odds change. Signed distractor deltas cancel by reciprocal pairing and are not used to claim no leakage.',
        numerical_checks=json.loads((RUN/'metrics.summary.json').read_text())['checks'])
    save('material_summary.json',summary)
    for name,rows in [('public_task_data.csv',public_table),('crossed_baseline_data.csv',cross_table),('patch_data.csv',patch_table),('lexical_block_data.csv',blocks)]:
        with (OUT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(overview=overview,example=summary['example'],primary_layer_totals=[dict(layer=l,correct=sum(r['correct'] for r in patch_table if r['model']=='1b' and r['layer']==l and r['site']=='changed_region_end' and r['factor']=='subject'),n=192) for l in [3,7,11]]),indent=2))

if __name__=='__main__':main()
