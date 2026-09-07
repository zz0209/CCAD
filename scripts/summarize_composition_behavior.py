"""Supplemental controls with common source references and explicit information."""
import argparse,collections,json,statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--prefix',default='R3');args=ap.parse_args();cfg=json.loads((args.run/'config.resolved.json').read_text());parent=Path(cfg['correspondence_run'])
    endpoints={}
    for path in parent.glob('pair_s*_t*.json'):
        for r in json.loads(path.read_text())['endpoint']:endpoints[(r['source_seed'],r['target_seed'],r['factor'],r['row_id'])]=r
    references={}
    for line in (parent/'metrics.raw.jsonl').open():
        r=json.loads(line)
        if r['method']=='source_teacher':references[(r['source_seed'],r['factor'],r['row_id'])]=r
    buckets=collections.defaultdict(list);checks=[];all_rows=0
    for line in (args.run/'metrics.raw.jsonl').open():
        r=json.loads(line);all_rows+=1;s=r['source_seed'];part='known_cue' if r['cue_id'] in cfg['discovery_cues'] else 'new_cue'
        if r['method']=='source_teacher':
            ref=references[(s,r['factor'],r['row_id'])];checks.append(max(abs(a-b) for a,b in zip(ref['label_logprobs'],r['label_logprobs']))<1e-9 and abs(ref['kl_reference']-r['kl_reference'])<1e-9);continue
        t=r.get('target_seed',next(seed['seed'] for seed in cfg['sae_checkpoints'] if seed['seed']!=s));e=endpoints[(s,t,r['factor'],r['row_id'])];r['noop_kl']=e['teacher_noop_kl'];r['teacher_agreement']=r['label']==e['teacher_label']
        for detail in [('partition',part),('source_partition',s,part)]+([('edge_partition',s,t,part)] if 'target_seed' in r else []):buckets[(r['factor'],r['method'])+detail].append(r)
    assert checks and all(checks),'Repeated source teacher readouts or full-donor KL differ across batches/runs'
    table=[]
    for key,rr in buckets.items():
        kl=sum(r['kl_reference'] for r in rr);noop=sum(r['noop_kl'] for r in rr)
        table.append(dict(factor=key[0],method=key[1],detail=list(key[2:]),n=len(rr),mean_kl=kl/len(rr),kl_sum=kl,noop_kl_sum=noop,kl_ratio=kl/noop,teacher_agreement=sum(r['teacher_agreement'] for r in rr)/len(rr),correct=sum(r['correct'] for r in rr),mean_norm=statistics.mean(r['delta_norm'] for r in rr)))
    gains=[]
    for path in args.run.glob('pair_s*_t*.json'):gains.extend(json.loads(path.read_text())['gain_fit'])
    result=dict(run=str(args.run),rows=table,raw_rows=all_rows,completed_gain_pairs=len(list(args.run.glob('pair_s*_t*.json'))),repeated_source_rows_checked=len(checks),repeated_source_checks_pass=True,gain_choices=gains,information=cfg['scope'],das_dependence='DAS has one fitted raw direction per source/factor, no target SAE. Its five source fits are not20independentseededges. Nativegain controls retain20dependentdirections.')
    args.out.mkdir(parents=True,exist_ok=True);(args.out/(args.prefix+'_BEHAVIOR_SUMMARY.json')).write_text(json.dumps(result,indent=2)+'\n')
    for r in table:
        if r['factor']=='joint' and r['detail'][0]=='partition':print(json.dumps(r))


if __name__=='__main__':main()
