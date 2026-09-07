"""Source-fixed full-denominator R3 summaries; seed edges stay dependent."""
import argparse,collections,json,statistics
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads((args.run/'config.resolved.json').read_text());base_rows=[json.loads(line) for line in (Path(cfg['material_run'])/'metrics.raw.jsonl').read_text().splitlines() if line]
    baseline={r['row_id']:r for r in base_rows if r['kind']=='baseline'};del base_rows
    endpoints={};diags=[]
    for path in sorted(args.run.glob('pair_s*_t*.json')):
        value=json.loads(path.read_text());diags.extend(value['diagnostics'])
        for r in value['endpoint']:endpoints[(r['source_seed'],r['target_seed'],r['factor'],r['row_id'])]=r
    buckets=collections.defaultdict(list);source_buckets=collections.defaultdict(list);cases={};count=0
    for line in (args.run/'metrics.raw.jsonl').open():
        r=json.loads(line);count+=1;part='known_cue' if r['cue_id'] in cfg['discovery_cues'] else 'new_cue';r['partition']=part
        if r['method']=='source_teacher':
            for detail in [('all',),('partition',part),('seed_partition',r['source_seed'],part)]:source_buckets[(r['factor'],)+detail].append(r)
        else:
            key=(r['source_seed'],r['target_seed'],r['factor'],r['row_id'])
            if key not in endpoints:continue # In-progress partial pair is not a completed result.
            e=endpoints[key];r['teacher_noop_kl']=e['teacher_noop_kl'];r['teacher_agreement']=r['label']==e['teacher_label']
            for detail in [('all',),('partition',part),('edge_partition',r['source_seed'],r['target_seed'],part),('block',r['block'],part)]:buckets[(r['factor'],r['method'])+detail].append(r)
        if r['method'] in cfg['single_factor_methods']+['source_teacher']:
            cases[(r['source_seed'],r.get('target_seed'),r['method'],r['row_id'],r['factor'])]=r
    def stats(rr):
        result=dict(n=len(rr),correct=sum(r['correct'] for r in rr),number_correct=sum((r['number_logodds']>0)==bool(r['expected']%2) for r in rr),time_correct=sum((r['past_logodds']>0)==bool(r['expected']//2) for r in rr),mean_kl=float(np.mean([r['kl_reference'] for r in rr])),mean_norm=float(np.mean([r['delta_norm'] for r in rr])))
        if 'teacher_noop_kl' in rr[0]:
            kl=sum(r['kl_reference'] for r in rr);noop=sum(r['teacher_noop_kl'] for r in rr)
            result.update(kl_sum=kl,noop_kl_sum=noop,kl_ratio=kl/max(noop,1e-15),teacher_agreement=sum(r['teacher_agreement'] for r in rr)/len(rr))
        return result
    table=[dict(factor=k[0],method=k[1],detail=list(k[2:]),**stats(v)) for k,v in buckets.items()]
    sources=[dict(factor=k[0],detail=list(k[1:]),**stats(v)) for k,v in source_buckets.items()]
    edges={(r['factor'],r['method'],*r['detail'][1:]):r for r in table if r['detail'][0]=='edge_partition'};wins=[]
    for factor in ['number','time','joint']:
        for part in ['known_cue','new_cue']:
            for method in sorted({r['method'] for r in table if r['factor']==factor and r['method']!='fcc_group'}):
                pairs=[(r,edges.get((factor,method,*key[2:]))) for key,r in edges.items() if key[0:2]==(factor,'fcc_group') and key[-1]==part]
                pairs=[(a,b) for a,b in pairs if b is not None]
                if pairs:wins.append(dict(factor=factor,partition=part,comparator=method,n_directions=len(pairs),fcc_lower_kl=sum(a['mean_kl']<b['mean_kl'] for a,b in pairs)))
    memberships=[]
    for target in [s['seed'] for s in cfg['sae_checkpoints']]:
        for factor in ['number','time']:
            groups=[r for r in diags if r['target_seed']==target and r['factor']==factor and r['method']=='fcc_group'];sets=[set(r['members']) for r in groups]
            if not sets:continue
            scores=[len(a&b)/len(a|b) for i,a in enumerate(sets) for b in sets[i+1:]]
            memberships.append(dict(target_seed=target,factor=factor,groups=[dict(source_seed=r['source_seed'],members=r['members']) for r in groups],all_source_core=sorted(set.intersection(*sets)),mean_pairwise_jaccard=statistics.mean(scores) if scores else None))
    interactions=[]
    for key,joint in cases.items():
        if key[-1]!='joint':continue
        prefix=key[:-1];number=cases.get(prefix+('number',));temporal=cases.get(prefix+('time',))
        if number is None or temporal is None:continue
        base=baseline[joint['row_id']];delta=[joint[k]-number[k]-temporal[k]+base[k] for k in ['number_logodds','past_logodds']]
        interactions.append(dict(source_seed=joint['source_seed'],target_seed=joint.get('target_seed'),method=joint['method'],row_id=joint['row_id'],partition=joint['partition'],number_interaction=delta[0],time_interaction=delta[1],separate_both_correct=bool(number['correct'] and temporal['correct']),joint_correct=joint['correct']))
    result=dict(run=str(args.run),raw_rows=count,completed_pairs=len(list(args.run.glob('pair_s*_t*.json'))),scope=cfg['scope'],rows=table,source_rows=sources,wins=wins,memberships=memberships,interaction_rows=interactions,statistics_note='All direction totals are descriptive;20edges share five seeds. No independent-edge CI or pvalue. Ratios sum KL before division. Expected time preference does not imply all other tenses ungrammatical.')
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'R3_COMPOSITION_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');(args.out/'R3_VECTOR_DIAGNOSTICS.json').write_text(json.dumps(dict(rows=diags),indent=2)+'\n')
    for r in table:
        if r['factor']=='joint' and r['detail'][0]=='partition':print(json.dumps(r))


if __name__=='__main__':main()
