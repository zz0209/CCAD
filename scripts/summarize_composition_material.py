import argparse,json,statistics,collections
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();rows=[json.loads(s) for s in (args.run/'metrics.raw.jsonl').read_text().splitlines()];table=[]
    cfg=json.loads((args.run/'config.resolved.json').read_text())
    for r in rows:
        known_lex=r['block'] in cfg.get('discovery_blocks',range(6));known_cue=r['cue_id'] in cfg.get('discovery_cues',[0,1])
        r['partition']=('known_lex' if known_lex else 'new_lex')+'_'+('known_cue' if known_cue else 'new_cue')
    for detail in [[],['cue_id'],['template'],['partition']]:
        buckets=collections.defaultdict(list)
        for r in rows:buckets[tuple(r.get(k) for k in ['kind','layer','seed','method','factor']+detail)].append(r)
        for key,rr in buckets.items():
            table.append(dict(**dict(zip(['kind','layer','seed','method','factor']+detail,key)),n=len(rr),correct=sum(r['correct'] for r in rr),number_correct=sum((r['number_logodds']>0)==bool(r['expected']%2) for r in rr),time_correct=sum((r['past_logodds']>0)==bool(r['expected']//2) for r in rr),mean_kl=statistics.mean(r.get('kl_to_full_donor',r.get('kl_reference',0)) for r in rr) if 'kl_to_full_donor' in rr[0] or 'kl_reference' in rr[0] else None))
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'r3_composition_material_summary.json').write_text(json.dumps(dict(source_run=str(args.run),rows=table),indent=2)+'\n')
    for r in table:
        if not any(k in r for k in ['template','cue_id','partition']):print(json.dumps(r))


if __name__=='__main__':main()
