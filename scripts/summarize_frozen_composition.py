"""All-case, dependency-aware descriptive summaries of frozen transfer."""
import argparse,json,csv
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();run=ROOT/a.run;out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((run/'metrics.summary.json').read_text());assert summary['status']=='PASS'
    cfg=json.loads((run/'config.resolved.json').read_text());panel=json.loads((ROOT/cfg['material_run']/'panel.json').read_text());rows=panel['rows'];teacher={};groups=defaultdict(lambda:defaultdict(float));examples={}
    for p in run.glob('source*_endpoint.json'):
        for r in json.loads(p.read_text())['rows']:teacher[r['source_seed'],r['factor'],r['row_id']]=r
    for line in (run/'metrics.raw.jsonl').open():
        r=json.loads(line);t=teacher[r['source_seed'],r['factor'],r['row_id']];part='familiar_syntax' if r['template'] in ['pp','subject_relative'] else r['template']
        for template in ['all',part,r['template']] if part!=r['template'] else ['all',part]:
            key=(template,r['factor'],r['method'],r['source_seed'],r.get('target_seed',0),r['block']);d=groups[key];d['n']+=1;d['correct']+=r['correct'];d['teacher_agree']+=r['label']==t['teacher_label'];d['kl']+=0. if r['method']=='source_teacher' else r['kl_reference'];d['source_donor_kl']+=r['kl_reference'] if r['method']=='source_teacher' else 0.;d['noop_kl']+=t['teacher_noop_kl'];d['number_logodds']+=r['number_logodds'];d['past_logodds']+=r['past_logodds']
        if r['source_seed']==1 and r.get('target_seed',2)==2 and r['block']==0 and r['cue_id']==0 and r['number']==0 and r['past']==0 and r['distractor']==0:
            examples[r['template'],r['factor'],r['method']]=dict(text=rows[r['row_id']]['text'],**r)
    block=[]
    for key,d in groups.items():block.append(dict(zip(['partition','factor','method','source_seed','target_seed','block'],key),**{k:int(v) if k in ['n','correct','teacher_agree'] else v for k,v in d.items()}))
    pair=defaultdict(lambda:defaultdict(float))
    for r in block:
        key=tuple(r[k] for k in ['partition','factor','method','source_seed','target_seed'])
        for k in ['n','correct','teacher_agree','kl','noop_kl','source_donor_kl']:pair[key][k]+=r[k]
    pair_rows=[]
    for key,d in pair.items():pair_rows.append(dict(zip(['partition','factor','method','source_seed','target_seed'],key),n=int(d['n']),correct=int(d['correct']),teacher_agree=int(d['teacher_agree']),accuracy=d['correct']/d['n'],teacher_agreement=d['teacher_agree']/d['n'],kl=d['kl']/d['n'],noop_kl=d['noop_kl']/d['n'],source_donor_kl=d['source_donor_kl']/d['n'] if key[2]=='source_teacher' else None))
    pooled=defaultdict(list)
    for r in pair_rows:pooled[r['partition'],r['factor'],r['method']].append(r)
    pooled_rows=[]
    for key,rr in pooled.items():
        n=sum(r['n'] for r in rr);ncorrect=sum(r['correct'] for r in rr);nagree=sum(r['teacher_agree'] for r in rr);sources=defaultdict(list)
        for r in rr:sources[r['source_seed']].append(r)
        seedmeans=[sum(r['kl']*r['n'] for r in ss)/sum(r['n'] for r in ss) for ss in sources.values()]
        pooled_rows.append(dict(zip(['partition','factor','method'],key),n=n,correct=ncorrect,teacher_agree=nagree,accuracy=ncorrect/n,teacher_agreement=nagree/n,kl=sum(r['kl']*r['n'] for r in rr)/n,source_donor_kl=sum(r['source_donor_kl']*r['n'] for r in rr)/n if key[2]=='source_teacher' else None,source_seed_kl_min=min(seedmeans),source_seed_kl_max=max(seedmeans),directions=len(rr),source_seed_means=seedmeans))
    comparisons=[]
    for part in ['all','familiar_syntax','pp','subject_relative','object_relative','subject_late']:
        fcc={(r['source_seed'],r['target_seed']):r for r in pair_rows if r['partition']==part and r['factor']=='joint' and r['method']=='fcc_group'}
        for method in sorted({r['method'] for r in pair_rows}):
            if method in ['fcc_group','source_teacher','das_style_raw_rank1']:continue
            other={(r['source_seed'],r['target_seed']):r for r in pair_rows if r['partition']==part and r['factor']=='joint' and r['method']==method};keys=fcc.keys()&other.keys()
            if keys:comparisons.append(dict(partition=part,method=method,fcc_lower_kl=sum(fcc[k]['kl']<other[k]['kl'] for k in keys),directions=len(keys),mean_difference=sum(fcc[k]['kl']-other[k]['kl'] for k in keys)/len(keys)))
    for name,rr in [('frozen_blocks',block),('frozen_pairs',pair_rows)]:
        with (out/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    result=dict(run=str(run),source_summary=summary,rows=pooled_rows,comparisons=comparisons,examples=list(examples.values()),statistics='Descriptive all-case means. Eight lexicalblocks andtwo cues;five sharedseeds. Twenty directions are dependent, not20independentreplicates. Seedrange uses meanover4targetseeds. No newmethod fit orrowselection. kl isKLfromfrozen-sourceoutputtocandidate,source itself0; original source raw kl was tofulldonor andis separately retainedassource_donor_kl.',frozen=True)
    (out/'R4_FROZEN_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([r for r in pooled_rows if r['factor']=='joint' and r['partition'] in ['familiar_syntax','object_relative','subject_late'] and r['method'] in ['source_teacher','fcc_group','full_code_ridge','rrr_rank4','dense_select','raw_native_units','same_members_native','direct_target_native','das_style_raw_rank1']],indent=2))


if __name__=='__main__':main()
