"""Summarize frozen BLiMP transfers without selecting on target outcomes."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse, csv, hashlib, json, statistics


def mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else None


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def analyze(run, out, partial=False):
    status = json.loads((run / 'status.json').read_text())
    if not partial:
        assert status['status'] == 'PASS', status
    config=json.loads((run/'config.resolved.json').read_text())
    root=Path(__file__).resolve().parents[1]
    dm=json.loads((root/config['data_manifest']).read_text())
    label_lookup={}
    for entry in dm['files']:
        path=Path(entry['path'])
        if path.suffix=='.jsonl' and path.stem in config['tasks']:
            for line in path.read_text().splitlines():
                row=json.loads(line)
                label_lookup[row['UID'],int(row['pairID'])]=(row.get('one_prefix_word_good',''),row.get('one_prefix_word_bad',''))
    grouped, probes, counts = defaultdict(list), defaultdict(list), Counter()
    with (run / 'metrics.raw.jsonl').open() as f:
        for line in f:
            row = json.loads(line)
            counts[row['kind']] += 1
            if row['kind'] == 'frozen_transfer':
                grouped[(row['objective'], row['seed'], row['selection_task'], row['task'], row['query'], row['method'])].append(row)
            elif row['kind'] == 'controlled_probe':
                probes[(row['query'], row['category'], row['method'])].append(row)
    results = []; lexical=[]
    for (obj, seed, selected_task, task, query, method), rr in grouped.items():
        if partial and len(rr) not in [64, 936]:
            continue
        assert len({r['row_id'] for r in rr}) == len(rr)
        assert len(rr) == (936 if task == selected_task else 64)
        changed = [r for r in rr if r['source_accuracy'] != r['clean_accuracy']]
        results.append(dict(objective=obj, source_seed=seed, target_seed=rr[0]['target_seed'], selection_task=selected_task,
            task=task, query=query, method=method, n=len(rr), split='positive' if task == selected_task else 'control',
            margin_mae=mean(r['margin_error'] for r in rr), clean_margin_mae=mean(abs(r['source_margin']-r['clean_margin']) for r in rr),
            clean_mean_margin=mean(r['clean_margin'] for r in rr), source_mean_margin=mean(r['source_margin'] for r in rr),
            edited_mean_margin=mean(r['margin'] for r in rr), mean_margin_decrement=mean(r['clean_margin']-r['margin'] for r in rr),
            clean_accuracy=mean(r['clean_accuracy'] for r in rr), source_accuracy=mean(r['source_accuracy'] for r in rr),
            edited_accuracy=mean(r['accuracy'] for r in rr), decision_agreement=mean(r['accuracy']==r['source_accuracy'] for r in rr),
            source_changed_decisions=len(changed), changed_decision_agreement=mean(r['accuracy']==r['source_accuracy'] for r in changed),
            source_kl=mean(r['source_kl'] for r in rr), source_effect_kl=mean(r['source_effect_kl'] for r in rr),
            edit_norm=mean(r['edit_norm'] for r in rr)))
        if task==selected_task and task.startswith('anaphor_'):
            by_label=defaultdict(list)
            for r in rr:
                by_label[label_lookup[task,r['row_id']]].append(r)
            for (good,bad),lr in by_label.items():
                lexical.append(dict(objective=obj,source_seed=seed,query=query,task=task,method=method,
                    good_word=good,bad_word=bad,n=len(lr),
                    mean_margin_decrement=mean(r['clean_margin']-r['margin'] for r in lr),
                    source_margin_decrement=mean(r['clean_margin']-r['source_margin'] for r in lr),
                    margin_mae=mean(r['margin_error'] for r in lr)))
    source_selections = json.loads((run/'source_selection.json').read_text())['selected']
    lookup = {(r['query'],r['selection_task'],r['task'],r['method']): r for r in results}
    source = []
    for selected in source_selections:
        key = (selected['query'],selected['selection_task'],selected['selection_task'],'source_group')
        if key not in lookup:
            continue
        row = lookup[key]
        controls = [v for k,v in lookup.items() if k[0]==key[0] and k[1]==key[1] and k[2]!=key[2] and k[3]=='source_group']
        if len(controls) != 2:
            continue
        source.append(dict(**row, dev_mean_margin_decrement=selected['mean_margin_decrement'],
            dev_selectivity=selected['source_selectivity_score'],
            held_selectivity=row['mean_margin_decrement']-mean(r['mean_margin_decrement'] for r in controls),
            held_control_decrement=mean(r['mean_margin_decrement'] for r in controls)))
    aggregates = []
    for obj in ['topk','matryoshka']:
        for method in ['source_group','group64','assignment64','assignment_one_scale','raw_full','raw_rank1','raw_rank2']:
            rr = [r for r in results if r['objective']==obj and r['method']==method and r['split']=='positive']
            if not rr:
                continue
            aggregates.append(dict(objective=obj, method=method, selections=len(rr), pairs=sum(r['n'] for r in rr),
                **{k:mean(r[k] for r in rr) for k in ['margin_mae','clean_margin_mae','mean_margin_decrement','clean_accuracy','source_accuracy','edited_accuracy','decision_agreement','source_kl','source_effect_kl']},
                source_changed_decisions=sum(r['source_changed_decisions'] for r in rr),
                changed_decision_agreement=(sum(r['changed_decision_agreement']*r['source_changed_decisions'] for r in rr if r['source_changed_decisions'])/sum(r['source_changed_decisions'] for r in rr)) if sum(r['source_changed_decisions'] for r in rr) else None,
                beats_unchanged_source=sum(r['margin_mae']<r['clean_margin_mae'] for r in rr)))
    comparisons = []
    for obj in ['topk','matryoshka']:
        for baseline in ['assignment64','assignment_one_scale','raw_full','raw_rank1','raw_rank2']:
            pairs = [(r,lookup[(r['query'],r['selection_task'],r['task'],baseline)]) for r in results if r['objective']==obj and r['method']=='group64' and r['split']=='positive' and (r['query'],r['selection_task'],r['task'],baseline) in lookup]
            if pairs:
                comparisons.append(dict(objective=obj,baseline=baseline,selections=len(pairs),group_wins=sum(a['margin_mae']<b['margin_mae'] for a,b in pairs),group_minus_baseline=mean(a['margin_mae']-b['margin_mae'] for a,b in pairs)))
    probe_summary=[]
    probe_rows=[]
    for (query,category,method),rr in probes.items():
        probe_rows.extend(rr)
        probe_summary.append(dict(query=query,category=category,method=method,n=len(rr),
            clean_response_probability=mean(r['clean_response_probability'] for r in rr),
            source_response_probability=mean(r['source_response_probability'] for r in rr),
            response_probability=mean(r['response_probability'] for r in rr),
            source_probability_decrement=mean(r['clean_response_probability']-r['source_response_probability'] for r in rr),
            probability_decrement=mean(r['clean_response_probability']-r['response_probability'] for r in rr),
            probability_mae=mean(abs(r['response_probability']-r['source_response_probability']) for r in rr),
            clean_probability_mae=mean(abs(r['clean_response_probability']-r['source_response_probability']) for r in rr),
            source_kl=mean(r['source_kl'] for r in rr),source_effect_kl=mean(r['source_effect_kl'] for r in rr)))
    if not partial:
        config = json.loads((run/'config.resolved.json').read_text())
        expected_probes = 1536 if config.get('run_controlled_probes', True) else 0
        assert len(source)==30 and len(results)==630 and len(probe_rows)==expected_probes
        expected_counts = dict(source_selection=23040, frozen_transfer=223440)
        if expected_probes:
            expected_counts['controlled_probe'] = expected_probes
        assert counts == expected_counts, counts
    out.mkdir(parents=True,exist_ok=True)
    data=dict(run=str(run),status=status,partial=partial,counts=dict(counts),source_selections=source,
        aggregates=aggregates,paired_comparisons=comparisons,query_metrics=results,lexical_contrasts=lexical,probe_summary=probe_summary,probe_rows=probe_rows,
        scope='Three original BLiMP paradigms;64 source-selection pairs and936 held pairs each. Controls use64 held pairs of each other paradigm. Five cyclic source edges share SAE seeds. No disjoint template, lexicon, or model claim.',
        inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in
            [run/n for n in ['metrics.raw.jsonl','source_selection.json','SOURCE_SELECTION_FREEZE.json','config.resolved.json','panel.json']]
            +[root/config['data_manifest']]+[Path(e['path']) for e in dm['files'] if Path(e['path']).suffix=='.jsonl' and Path(e['path']).stem in config['tasks']]])
    write_json(out/'summary.json',data)
    for name,rows in [('query_metrics',results),('source_selections',source),('lexical_contrasts',lexical),('probe_summary',probe_summary)]:
        if rows:
            with (out/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--partial',action='store_true');a=p.parse_args()
    d=analyze(a.run,a.out,a.partial)
    print(json.dumps({k:d[k] for k in ['counts','aggregates','paired_comparisons']},indent=2))
