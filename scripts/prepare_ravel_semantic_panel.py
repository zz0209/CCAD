"""Deterministically pair known original-train cities, before interventions."""
from __future__ import annotations
import hashlib
import json
import argparse
import copy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CAP=ROOT/'runs/FINAL5_R16_ravel_capability_v1_20260908'
OUT=ROOT/'artifacts/final_five_research_20260908/r16_semantic_operations'
TEMPLATES={'Country':[27,39],'Continent':[5,14],'Language':[3,23]}
SALT='r16-independent-city-pairs-v1'

def expand_fit_pairings(donors_per_entity=8):
    """Increase donor diversity inside the existing fitting entity partition."""
    source=OUT/'semantic_panel_v1.json';original=json.loads(source.read_text())
    old=original['rows'];fit=[r for r in old if r['split']=='fit']
    entities=sorted({r['entity'] for r in fit});lookup={(r['entity'],r['task'],r['template_index']):r for r in fit}
    digest=lambda s:hashlib.sha256(('r16-diverse-fit-donors-v2|'+s).encode()).hexdigest()
    selected={tuple(sorted([r['entity'],r['donor_entity']])) for r in fit}
    inventory=[]
    for base in entities:
        candidates=[donor for donor in entities if donor!=base and all(
            set(lookup[base,a,t]['expected_ids']).isdisjoint(lookup[donor,a,t]['expected_ids'])
            for a,tt in TEMPLATES.items() for t in tt)]
        chosen=sorted(candidates,key=lambda d:digest(base+'|'+d))[:donors_per_entity]
        inventory.append(dict(entity=base,compatible_donors=len(candidates),selected=chosen))
        selected.update(tuple(sorted([base,donor])) for donor in chosen)
    rows=[]
    for a,b in sorted(selected,key=lambda x:digest('|'.join(x))):
        component='fit_cross_pair_'+digest(a+'|'+b)[:16]
        for attr,tt in TEMPLATES.items():
            for ti in tt:
                start=len(rows)
                for base,donor,di in [(a,b,start+1),(b,a,start)]:
                    r=copy.deepcopy(lookup[base,attr,ti]);d=lookup[donor,attr,ti]
                    r.update(row_id=len(rows),donor_id=di,component=component,pair_key=component+'_'+attr+'_'+str(ti),
                        donor_entity=donor,donor_label=d['label'],donor_expected_ids=d['expected_ids'],
                        donor_capability_correct=d['base_capability_correct'],source_panel_row_id=r['row_id'])
                    rows.append(r)
    offset=len(rows)
    held=[r for r in old if r['split']!='fit'];old_to_new={r['row_id']:offset+j for j,r in enumerate(held)}
    for r in held:
        new=copy.deepcopy(r);new.update(row_id=old_to_new[r['row_id']],donor_id=old_to_new[r['donor_id']],source_panel_row_id=r['row_id']);rows.append(new)
    path=OUT/'semantic_panel_v2_fit_crosspairs.json'
    if path.exists():raise FileExistsError(path)
    result=dict(rows=rows,selection_inventory=dict(original_panel=str(source),original=original['selection_inventory'],
        fit_entities=len(entities),fit_pairs=len(selected),fit_donor_choices=inventory,donors_per_entity=donors_per_entity,
        scope='Fit donor pairs expanded only within the original48fitting cities, preserving all original fit pairs. Calibration and held city pairs, templates and labels remain exactly the same. All exposed panels remain development; expanded training pairs sharing cities are not independent units.'),
        source_files=original['source_files']+[str(source)])
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(path=str(path),rows=len(rows),fit_pairs=len(selected),fit_entities=len(entities),calibration_and_held_rows=len(held))))

def main():
    from transformers import AutoTokenizer
    cfg=json.loads((CAP/'config.resolved.json').read_text())
    tok=AutoTokenizer.from_pretrained(cfg['model_local_dir'],local_files_only=True,trust_remote_code=False)
    knowledge=json.loads((CAP/'entity_knowledge.json').read_text())['entities']
    known=[e for e,a in knowledge.items() if all(a.get(k,{}).get('correct',0)>=2 for k in TEMPLATES)]
    panel=json.loads((CAP/'panel.json').read_text())['rows']
    lookup={(r['entity'],r['attribute'],r['template_index']):r for r in panel}
    scores={(r['entity'],r['task'],r['template_index']):r for r in map(json.loads,(CAP/'metrics.raw.jsonl').read_text().splitlines())}
    digest=lambda s:hashlib.sha256((SALT+s).encode()).hexdigest()
    eligible=[e for e in known if all((e,a,t) in lookup for a,tt in TEMPLATES.items() for t in tt)]
    candidates=[]
    for j,a in enumerate(eligible):
        for b in eligible[j+1:]:
            if all(set(scores[a,attr,t]['expected_first_token_ids']).isdisjoint(scores[b,attr,t]['expected_first_token_ids'])
                   for attr,tt in TEMPLATES.items() for t in tt):
                candidates.append((digest('|'.join(sorted([a,b]))),a,b))
    pairs=[];used=set()
    for key,a,b in sorted(candidates):
        if a in used or b in used:continue
        pairs.append((key,a,b));used.update([a,b])
        if len(pairs)==40:break
    if len(pairs)!=40:raise ValueError(f'Only {len(pairs)} disjoint structurally compatible pairs')
    rows=[];pair_inventory=[]
    for pi,(key,a,b) in enumerate(sorted(pairs,key=lambda x:digest('split'+x[0]))):
        split='fit' if pi<24 else 'calibration' if pi<32 else 'held_component_development'
        component='city_pair_'+key[:16]
        pair_inventory.append(dict(component=component,entities=[a,b],split=split))
        for attr,tt in TEMPLATES.items():
            for ti in tt:
                start=len(rows)
                for entity,donor,donor_id in [(a,b,start+1),(b,a,start)]:
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
                    rows.append(dict(row_id=len(rows),text=old['text'],spans=spans,label=label,donor_label=donor_label,
                        donor_id=donor_id,task=attr,attribute=attr,entity=entity,donor_entity=donor,
                        component=component,pair_key=component+'_'+attr+'_'+str(ti),split=split,
                        changed_region=1,template_index=ti,expected_ids=s['expected_first_token_ids'],
                        donor_expected_ids=ds['expected_first_token_ids'],base_capability_correct=s['first_token_correct'],
                        donor_capability_correct=ds['first_token_correct'],original_entity_split='train',original_template_split='train'))
    inventory=dict(original_train_entities=len(knowledge),knowledge_eligible=len(known),structurally_complete=len(eligible),
        compatible_candidate_pairs=len(candidates),selected_disjoint_pairs=pair_inventory,selected_entities=sorted(used),
        known_unused_entities=sorted(set(known)-used),templates=TEMPLATES,salt=SALT,
        selection='At least2of3 first-token answers correct per attribute; every selected template exists; disjoint expected first-token alias sets for all three attributes; hash-greedy disjoint pairs. No intervention or SAE outcomes.',
        scope='Original train entities/templates already screened. Held pairs are disjoint development entities, not untouched official validation/test or a frozen scientific confirmation.',
        labels='First accepted unedited top1 when correct, otherwise first expected token. Main accuracy accepts all expected first-token aliases; no full-answer generation claim.')
    sources=[str(CAP/n) for n in ['config.resolved.json','entity_knowledge.json','panel.json','metrics.raw.jsonl']]
    output=dict(rows=rows,selection_inventory=inventory,source_files=sources)
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'semantic_panel_v1.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(rows=len(rows),pairs=len(pairs),known=len(known),compatible_pairs=len(candidates),path=str(path))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--expand-fit',action='store_true');args=p.parse_args()
    expand_fit_pairings() if args.expand_fit else main()
