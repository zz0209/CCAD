"""Pair all possible reserved cities using metadata, without interventions."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from prepare_ravel_semantic_panel import CAP,TEMPLATES


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import networkx as nx
    from transformers import AutoTokenizer
    original_path=Path('artifacts/final_five_research_20260908/r17_projected_semantics/semantic_panel_v3_176cities.json')
    original=json.loads(original_path.read_text());cities=original['selection_inventory']['reserved_no_intervention_entities']
    cfg=json.loads((CAP/'config.resolved.json').read_text());tok=AutoTokenizer.from_pretrained(cfg['model_local_dir'],local_files_only=True,trust_remote_code=False)
    panel=json.loads((CAP/'panel.json').read_text())['rows'];lookup={(r['entity'],r['attribute'],r['template_index']):r for r in panel}
    scores={(r['entity'],r['task'],r['template_index']):r for r in map(json.loads,(CAP/'metrics.raw.jsonl').read_text().splitlines())}
    salt='r18-reserved-city-maximum-cardinality-20260908';digest=lambda s:hashlib.sha256((salt+'|'+s).encode()).hexdigest()
    graph=nx.Graph();graph.add_nodes_from(sorted(cities))
    for j,a in enumerate(sorted(cities)):
        for b in sorted(cities)[j+1:]:
            if all(set(scores[a,t,i]['expected_first_token_ids']).isdisjoint(scores[b,t,i]['expected_first_token_ids']) for t,ts in TEMPLATES.items() for i in ts):
                graph.add_edge(a,b,weight=int(digest(a+'|'+b),16))
    # Maximum cardinality first; fixed hash weights choose among equal-size matchings.
    pairs=sorted([tuple(sorted(p)) for p in nx.max_weight_matching(graph,maxcardinality=True)],key=lambda p:digest('order|'+'|'.join(p)))
    rows=copy.deepcopy(original['rows']);old_entities={r['entity'] for r in rows};used={e for pair in pairs for e in pair}
    assert not old_entities&set(cities)
    for a,b in pairs:
        component='confirm_city_pair_'+digest(a+'|'+b)[:16]
        for task,templates in TEMPLATES.items():
            for ti in templates:
                start=len(rows)
                for entity,donor,donor_id in [(a,b,start+1),(b,a,start)]:
                    old=lookup[entity,task,ti];s=scores[entity,task,ti];ds=scores[donor,task,ti]
                    choose=lambda r:r['top_ids'][0] if r['first_token_correct'] else r['expected_first_token_ids'][0]
                    prefix,suffix=old['template'].split('%s');trim=prefix.rstrip(' ');spans=[trim,prefix[len(trim):]+entity,suffix]
                    tokens=tok.encode(old['text'],add_special_tokens=False);acc=''
                    for span in spans:
                        acc+=span;pt=tok.encode(acc,add_special_tokens=False)
                        if tokens[:len(pt)]!=pt:raise ValueError('Invalid token region boundary')
                    label,donor_label=tok.decode([choose(s)]),tok.decode([choose(ds)])
                    for lab in [label,donor_label]:
                        if tok.encode(old['text']+lab,add_special_tokens=False)!=tokens+tok.encode(lab,add_special_tokens=False):raise ValueError('Invalid label boundary')
                    rows.append(dict(row_id=len(rows),text=old['text'],spans=spans,label=label,donor_label=donor_label,
                        donor_id=donor_id,task=task,attribute=task,entity=entity,donor_entity=donor,component=component,
                        pair_key=f'{component}_{task}_{ti}',split='confirmation',changed_region=1,template_index=ti,
                        expected_ids=s['expected_first_token_ids'],donor_expected_ids=ds['expected_first_token_ids'],
                        base_capability_correct=s['first_token_correct'],donor_capability_correct=ds['first_token_correct'],
                        original_entity_split='train',original_template_split='train'))
    for r in rows:
        d=rows[r['donor_id']]
        assert d['donor_id']==r['row_id'] and d['entity']==r['donor_entity'] and d['split']==r['split']
    inputs=[original_path,CAP/'config.resolved.json',CAP/'panel.json',CAP/'metrics.raw.jsonl']
    inventory=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),reserved_entities=cities,selected_pairs=pairs,
        reserved_entities_used=sorted(used),unused_entities=sorted(set(cities)-used),compatible_edges=graph.number_of_edges(),
        maximum_disjoint_pairs=len(pairs),confirmation_rows=len(rows)-len(original['rows']),salt=salt,networkx_version=nx.__version__,
        source_panel=original['selection_inventory'],inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],
        selection='Maximum-cardinality general graph matching with fixed hash weights; original all-attribute alias-disjoint metadata only. No edited source/target outcomes used.',
        scope='New cities for intervention confirmation within the earlier unedited knowledge-screened original RAVEL train population. Same original task templates; not official benchmark test or a new model. Old fit/cal/held retained verbatim to replay source identities. No intervention has been performed on these new rows by this preparation.')
    result=dict(rows=rows,selection_inventory=inventory,source_files=original['source_files']+[str(original_path)])
    args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(args.output),pairs=len(pairs),cities=len(used),unused=len(cities)-len(used),rows=len(rows),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
