"""Descriptive dependence-aware summaries for frozen predictive-member operations."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    run=args.run;cfg=json.loads((run/'config.resolved.json').read_text());summary=json.loads((run/'metrics.summary.json').read_text());assert summary['status']=='PASS'
    out=ROOT/'artifacts/seven_round_rebuild_20260906/r5_mechanism';out.mkdir(exist_ok=True)
    panel=json.loads((ROOT/cfg['material_run']/'panel.json').read_text());rows=panel['rows']
    baseline={}
    for line in (ROOT/cfg['material_run']/'metrics.raw.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['kind']=='baseline':baseline[row['row_id']]=row
    groups={};full={};examples=[];count=0
    metrics=['kl','accuracy','number_shift','time_shift','abs_number_shift','abs_time_shift','number_difference_from_fcc','time_difference_from_fcc','abs_number_difference_from_fcc','abs_time_difference_from_fcc','norm']
    for line in (run/'metrics.raw.jsonl').open():
        row=json.loads(line);count+=1;i=row['row_id'];p=rows[i];base=baseline[i];identity=(row['source_seed'],row['target_seed'],row['factor'],i)
        if row['method']=='fcc_group':full[identity]=row
        intact=full.get(identity,row);dn=row['number_logodds']-base['number_logodds'];dt=row['past_logodds']-base['past_logodds']
        orientation_n=1 if rows[panel['pairs'][i]['number']]['number'] else -1
        orientation_t=1 if rows[panel['pairs'][i]['time']]['past'] else -1
        values=[row['kl_reference'],float(row['correct']),dn*orientation_n,dt*orientation_t,abs(dn),abs(dt),
            (row['number_logodds']-intact['number_logodds'])*orientation_n,(row['past_logodds']-intact['past_logodds'])*orientation_t,
            abs(row['number_logodds']-intact['number_logodds']),abs(row['past_logodds']-intact['past_logodds']),row['delta_norm']]
        role=p['cue_role'];cue='familiar_cue' if p['cue_id']<2 else 'new_cue'
        for part in [role,role+'/'+cue,role+'/'+p['template']+'/'+cue,role+'/block'+str(p['block'])]:
            key=(part,row['factor'],row['method'],row['source_seed'],row['target_seed'],row.get('member',-1),row['reference_kind'])
            if key not in groups:groups[key]=[0,np.zeros(len(metrics))]
            groups[key][0]+=1;groups[key][1]+=values
        if row['source_seed']==1 and row['target_seed']==2 and p['block']==0 and p['number']==p['past']==p['distractor']==0:
            examples.append(dict(text=p['text'],donor_text=rows[panel['pairs'][i][row['factor']]]['text'],cue_role=role,baseline=base,**row))
    assert count==summary['rows']
    keys=['partition','factor','method','source_seed','target_seed','member','reference_kind']
    detail=[dict(zip(keys,key),n=n,**dict(zip(metrics,(total/n).tolist()))) for key,(n,total) in groups.items()]
    # Every direction gets the same weight, after averaging over its own members.
    direction_groups=defaultdict(list)
    for row in detail:direction_groups[tuple(row[k] for k in keys if k!='member')].append(row)
    directions=[]
    direction_keys=[k for k in keys if k!='member']
    for key,rr in direction_groups.items():
        directions.append(dict(zip(direction_keys,key),members=len(rr),n_per_member=rr[0]['n'],**{k:float(np.mean([row[k] for row in rr])) for k in metrics}))
    pooled_groups=defaultdict(list)
    for row in directions:pooled_groups[row['partition'],row['factor'],row['method'],row['reference_kind']].append(row)
    pooled=[]
    for key,rr in pooled_groups.items():
        pooled.append(dict(zip(['partition','factor','method','reference_kind'],key),directions=len(rr),
            **{k:float(np.mean([row[k] for row in rr])) for k in metrics},source_seed_kl_min=min(row['kl'] for row in rr),source_seed_kl_max=max(row['kl'] for row in rr)))
    lookup={(r['partition'],r['factor'],r['method'],r['source_seed'],r['target_seed'],r['member']):r for r in detail}
    member_effects=[]
    for row in detail:
        if row['method']!='without_member':continue
        control=lookup[row['partition'],row['factor'],'without_member_norm_control',row['source_seed'],row['target_seed'],row['member']]
        member_effects.append(dict(partition=row['partition'],factor=row['factor'],source_seed=row['source_seed'],target_seed=row['target_seed'],member=row['member'],n=row['n'],
            deletion_kl=row['kl'],size_control_kl=control['kl'],excess_kl=row['kl']-control['kl'],deletion_accuracy=row['accuracy'],
            signed_number_loss=-row['number_difference_from_fcc'],signed_time_loss=-row['time_difference_from_fcc']))
    geometry=json.loads((run/'member_decomposition.json').read_text());gl={(r['source_seed'],r['target_seed'],r['factor'],r['role']):r for r in geometry['rows']}
    ratios=[]
    for s in range(1,6):
        for t in range(1,6):
            if s==t:continue
            for factor in ['number','time']:
                q=gl[s,t,factor,'quoted'];a=gl[s,t,factor,'temporal']
                ratios.append(dict(source_seed=s,target_seed=t,factor=factor,individual_energy_ratio=q['individual_energy']/a['individual_energy'],total_energy_ratio=q['total_energy']/a['total_energy'],
                    cross_term_multiplier=q['total_to_individual_energy']/a['total_to_individual_energy'],temporal_cross_energy=a['cross_energy'],quoted_cross_energy=q['cross_energy']))
    for name,records in [('member_effects',member_effects),('mechanism_directions',directions),('mechanism_member_rows',detail)]:
        with (out/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    result=dict(run=str(run),run_summary=summary,rows=pooled,geometry_ratios=ratios,memberships=geometry['memberships'],fixed_examples=examples,checks=json.loads((run/'mechanism_checks.json').read_text()),
        raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest(),
        statistics='All512exposedprompts. Geometry20dependentdirections; interventions5preselectedcycledirections. Pooldirections equally after own-member means, noindependent-edge ormember confidenceinterval. Native/swaps/member KL refer to intactFCC; FCC KL refers to source; source KL to full donor. Differentreferencekindsnevercombined. Allfixedexamples source1target2block0initialvalues0, notselectedforcorrectness.')
    (out/'R5_MECHANISM_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in pooled:
        if '/' not in row['partition'] and row['method'] in ['fcc_group','native_norm_matched','role_swap','role_swap_norm_matched','joint_time_role_swap','joint_time_role_swap_norm_matched','without_member','without_member_norm_control']:
            print(json.dumps({k:row[k] for k in ['partition','factor','method','reference_kind','kl','accuracy','time_shift','number_shift']}))


if __name__=='__main__':main()
