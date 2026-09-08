"""Increase fitting-city coverage at a fixed paired-row budget.

Uses the original unedited capability screen only. Original calibration and
held-development pairs are copied unchanged; unused cities remain reserved.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from prepare_ravel_semantic_panel import CAP, OUT, ROOT, TEMPLATES


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--additional-cities',type=int,default=128)
    ap.add_argument('--fit-pairs',type=int,default=351)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    from transformers import AutoTokenizer
    source=OUT/'semantic_panel_v1.json';original=json.loads(source.read_text())
    cfg=json.loads((CAP/'config.resolved.json').read_text())
    tok=AutoTokenizer.from_pretrained(cfg['model_local_dir'],local_files_only=True,trust_remote_code=False)
    salt='r17-city-diversity-fixed-pair-budget-v1'
    digest=lambda s:hashlib.sha256((salt+'|'+s).encode()).hexdigest()
    unused=original['selection_inventory']['known_unused_entities']
    added=sorted(unused,key=lambda s:digest('newfit|'+s))[:args.additional_cities]
    if len(added)!=args.additional_cities:raise ValueError('Insufficient unused cities')
    original_fit=[r for r in original['rows'] if r['split']=='fit']
    old_entities={r['entity'] for r in original_fit}
    entities=sorted(old_entities|set(added))
    panel=json.loads((CAP/'panel.json').read_text())['rows']
    lookup={(r['entity'],r['attribute'],r['template_index']):r for r in panel}
    scores={(r['entity'],r['task'],r['template_index']):r for r in
            map(json.loads,(CAP/'metrics.raw.jsonl').read_text().splitlines())}
    def compatible(a,b):
        return all(set(scores[a,attr,t]['expected_first_token_ids']).isdisjoint(
            scores[b,attr,t]['expected_first_token_ids']) for attr,tt in TEMPLATES.items() for t in tt)
    candidates=sorted([(a,b) for j,a in enumerate(entities) for b in entities[j+1:] if compatible(a,b)],
                      key=lambda p:digest('pair|'+'|'.join(p)))
    selected={tuple(sorted([r['entity'],r['donor_entity']])) for r in original_fit}
    covered={e for p in selected for e in p}
    # Include every fitting city before filling the remaining fixed budget.
    for a,b in candidates:
        if a not in covered or b not in covered:
            selected.add((a,b));covered.update([a,b])
    if covered!=set(entities):raise ValueError('A fitting city has no compatible donor')
    if len(selected)>args.fit_pairs:raise ValueError('Coverage exceeds declared pair budget')
    for pair in candidates:
        if len(selected)==args.fit_pairs:break
        selected.add(pair)
    if len(selected)!=args.fit_pairs:raise ValueError('Insufficient compatible pairs')
    rows=[]
    def make_row(entity,donor,attr,ti,component,donor_id):
        old=lookup[entity,attr,ti];s=scores[entity,attr,ti];ds=scores[donor,attr,ti]
        choose=lambda r:r['top_ids'][0] if r['first_token_correct'] else r['expected_first_token_ids'][0]
        prefix,suffix=old['template'].split('%s');trimmed=prefix.rstrip(' ')
        spans=[trimmed,prefix[len(trimmed):]+entity,suffix]
        tokens=tok.encode(old['text'],add_special_tokens=False);acc=''
        for span in spans:
            acc+=span;pt=tok.encode(acc,add_special_tokens=False)
            if tokens[:len(pt)]!=pt:raise ValueError('Invalid token region boundary')
        label,donor_label=tok.decode([choose(s)]),tok.decode([choose(ds)])
        for lab in [label,donor_label]:
            if tok.encode(old['text']+lab,add_special_tokens=False)!=tokens+tok.encode(lab,add_special_tokens=False):
                raise ValueError('Invalid label boundary')
        return dict(row_id=len(rows),text=old['text'],spans=spans,label=label,donor_label=donor_label,
            donor_id=donor_id,task=attr,attribute=attr,entity=entity,donor_entity=donor,
            component=component,pair_key=component+'_'+attr+'_'+str(ti),split='fit',
            changed_region=1,template_index=ti,expected_ids=s['expected_first_token_ids'],
            donor_expected_ids=ds['expected_first_token_ids'],base_capability_correct=s['first_token_correct'],
            donor_capability_correct=ds['first_token_correct'],original_entity_split='train',original_template_split='train')
    for a,b in sorted(selected,key=lambda p:digest('order|'+'|'.join(p))):
        component='diverse_fit_pair_'+digest(a+'|'+b)[:16]
        for attr,tt in TEMPLATES.items():
            for ti in tt:
                start=len(rows)
                rows.append(make_row(a,b,attr,ti,component,start+1))
                rows.append(make_row(b,a,attr,ti,component,start))
    held=[r for r in original['rows'] if r['split']!='fit']
    mapping={r['row_id']:len(rows)+j for j,r in enumerate(held)}
    for old in held:
        row=copy.deepcopy(old);row.update(row_id=mapping[old['row_id']],donor_id=mapping[old['donor_id']],source_panel_row_id=old['row_id'])
        rows.append(row)
    nonfit={r['entity'] for r in held}
    reserved=sorted(set(unused)-set(added))
    if set(entities)&nonfit or (set(entities)|nonfit)&set(reserved):raise ValueError('Entity partition overlap')
    for r in rows:
        d=rows[r['donor_id']]
        if d['donor_id']!=r['row_id'] or d['entity']!=r['donor_entity'] or d['task']!=r['task']:
            raise ValueError('Reciprocal pairing differs')
    inputs=[]
    for path in [source,CAP/'config.resolved.json',CAP/'panel.json',CAP/'metrics.raw.jsonl']:
        inputs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    inventory=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),salt=salt,fit_entities=entities,
        original_fit_entities=sorted(old_entities),added_fit_entities=added,fit_pairs=len(selected),
        original_fit_pairs_retained=len({tuple(sorted([r['entity'],r['donor_entity']])) for r in original_fit}),
        calibration_and_held_entities=sorted(nonfit),reserved_no_intervention_entities=reserved,
        fit_city_pair_degrees={e:sum(e in p for p in selected) for e in entities},inputs=inputs,
        selection='Original unedited capability criterion plus deterministic hash selection, full city coverage then fixed351pair budget. No SAE or intervention outputs used.',
        scope='More fitting cities at the same4212fit rows as the48city expanded panel. Old96cal/96held rows copied unchanged; these remain development. Reserved cities passed only the previous unedited screen and have no intervention outcomes in this panel. No official test-set or generalization claim.')
    result=dict(rows=rows,selection_inventory=inventory,source_files=[r['path'] for r in inputs])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(args.output),fit_cities=len(entities),fit_pairs=len(selected),rows=len(rows),
                         reserved_cities=len(reserved),sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__=='__main__':main()
