from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/final_science_20260920'


def save(path,value):
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=json.loads((ROOT/'configs/intervention_source_metric_budget_20260920.json').read_text())
    original=json.loads((ROOT/'configs/reform_r58_shift_source_development_v2.json').read_text())
    prior_path=ROOT/'artifacts/independent_reuse_20260916/IR04_NEW_TASKS_PANEL.json'
    prior=json.loads(prior_path.read_text())
    paths=[prior_path,Path(base['frozen_source_run'])/'panel.json',
           ROOT/'artifacts/science_upgrade_20260919/ROUND04_HUMAN_PANEL.json',
           ROOT/'artifacts/transport_exploration_20260919/FRESH_HUMAN_PANEL.json']
    paths.extend(ROOT/r['path'] for r in prior['exclusion_sources'])
    excluded=set(); provenance=[]
    for path in dict.fromkeys(paths):
        data=json.loads(path.read_text())
        excluded.update(r['document_sha256'] for r in data['rows'])
        provenance.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    cells={(p,g):[] for p in [5,25,12,24] for g in [0,1]}
    table=pq.read_table(original['dev_data'],filters=[('profession','in',[5,25,12,24])],columns=['hard_text','profession','gender'])
    for r in table.to_pylist():
        digest=hashlib.sha256(r['hard_text'].encode()).hexdigest()
        if digest in excluded: continue
        excluded.add(digest)
        cells[r['profession'],r['gender']].append(dict(text=r['hard_text'],document_sha256=digest,profession=r['profession'],gender=r['gender'],split='confirm'))
    rows=[]
    for cell,values in cells.items():
        if len(values)<16: raise ValueError(f'Insufficient unseen documents in {cell}')
        rows+=sorted(values,key=lambda r:hashlib.sha256(('profile01/'+r['document_sha256']).encode()).hexdigest())[:16]
    tok=AutoTokenizer.from_pretrained(base['model_local_dir'],local_files_only=True)
    for i,r in enumerate(rows): r.update(row_id=i,tokens=tok.encode(r['text'],truncation=True,max_length=2048))
    human_path=OUT/'PROFILE_CONFIRMATION_HUMAN.json'
    save(human_path,dict(rows=rows,original_split='dev',exclusions=provenance,counts_before_selection={str(k):len(v) for k,v in cells.items()}))
    grammar_paths=['artifacts/independent_reuse_20260916/IR01_fit_panel.json','artifacts/independent_reuse_20260916/IR01_evaluation_panel.json',
                   'artifacts/science_upgrade_20260919/ROUND03_INFINITIVE_PANEL.json','artifacts/science_upgrade_20260919/ROUND04_INFINITIVE_PANEL.json',
                   'artifacts/transport_exploration_20260919/FRESH_GRAMMAR_PANEL.json']
    previous=[r for path in grammar_paths for r in json.loads((ROOT/path).read_text())['rows']]
    oldv={r['pair'].split(':')[0] for r in previous}; oldn={r['pair'].split(':')[1] for r in previous}
    verbs=['hired','paid','selected','scheduled','recruited','commissioned']; nouns=['architects','chemists','editors','singers']
    if set(verbs)&oldv or set(nouns)&oldn: raise ValueError('Grammatical vocabulary has prior exposure')
    grows=[]
    for verb in verbs:
        for noun in nouns:
            forms=[f'The {noun} had been {verb}',f'The committee {verb} the {noun}',f'Yesterday the {noun} were {verb}',f'At the meeting, the supervisor {verb} the {noun}']
            for form,text in enumerate(forms): grows.append(dict(pair=f'{verb}:{noun}',verb=verb,noun=noun,form=form,role='predicate' if form%2==0 else 'object',text=text))
    rng=np.random.default_rng(2026092001); groups=['pronouns','names','associated_words']
    hq=base['human_queries']; families={q:'endpoints' if q!='center' else 'center' for q in hq}
    gq={f'members_{bits:04b}':[(bits>>i)&1 for i in range(4)] for bits in range(1,16)}
    gf={q:'endpoints' if v in [[1,1,1,0],[0,0,0,1],[1,1,1,1]] else 'members' for q,v in gq.items()}
    for j in range(8):
        name=f'participation_{j:02d}'; weights=rng.uniform(.1,.9,3)
        if j>=4: weights[j%3]=j%2
        hq[name]=dict(zip(groups,weights.tolist())); families[name]='participation'
        pair=rng.uniform(.1,.9,2)
        if j>=4: pair[j%2]=(j//2)%2
        gq[name]=[float(pair[0])]*3+[float(pair[1])]; gf[name]='participation'
    source=json.loads(Path(base['source_manifest']).read_text())
    masks={}
    for j in range(4):
        name=f'member_subset_{j:02d}'; masks[name]={s:rng.integers(0,2,len(ids)).tolist() for s,ids in source['members'].items()}; families[name]='members'
    grammar_path=OUT/'PROFILE_CONFIRMATION_GRAMMAR.json'
    save(grammar_path,dict(rows=grows,queries=gq,families=gf,answer=' to',previous_panels=grammar_paths))
    request_path=OUT/'PROFILE_CONFIRMATION_REQUESTS.json'
    save(request_path,dict(human_queries=hq,human_member_queries=masks,human_families=families,grammar_families=gf,random_seed=2026092001))
    configurations=[]
    for seed in range(1,6):
        c={**base,'run_id':f'PROFILE_CONFIRMATION_T{seed}_20260920','target_seed':seed,'seeds':[seed],
           'human_panel':str(human_path),'human_split':'confirm','human_per_cell':16,'grammar_panel':str(grammar_path),'grammar_rows':96,
           'human_queries':hq,'human_member_queries':masks,'request_panel':str(request_path),'eval_token_budget':2048,
           'audit_opened':True,'candidate_family_frozen':True,'evidence_level':'frozen_fresh_context_request_confirmation',
           'purpose':'Reuse one frozen source sensitivity profile across five target dictionaries and new part requests',
           'scope':'Two fixed published source explanations; new biographies and grammatical vocabulary; four fixed later heads excluded from the profile; target dictionaries and source profile unchanged.',
           'statistics_unit':'Paired target seeds, stratified profession/gender documents, crossed verb/noun units, request families and fixed later heads.',
           'budget':'Five drivers at most900seconds each; original2p common member budget, no fitting; same frozen profile shared across seeds.',
           'budget_seconds':900,'query_progress':True}
        if seed==1:
            c['target_directory']='D:/CCAD_Storage/training_curves/REFORM_R58_shift_dictionaries_curve_v1_20260915/step_8192'
            c.pop('task_adapted_reference')
        else:
            c['task_adapted_reference']={'human':{s:f'D:/CCAD_Storage/training_curves/SCIENCE04_shift_t{seed}_v1_20260919/tangent_mixed/{s}_seed{seed}.pt' for s in ['embed','mlp_0','resid_0']},
                'grammar':base['task_adapted_reference']['grammar'] if seed==2 else f'D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE04_infinitive_t{seed}_v1_20260919/tangent_mixed/dictionary.pt'}
            for path in list(c['task_adapted_reference']['human'].values())+[c['task_adapted_reference']['grammar']]:
                if not Path(path).is_file(): raise FileNotFoundError(path)
        path=ROOT/f'configs/profile_confirmation_t{seed}_20260920.json'; save(path,c); configurations.append(str(path))
    freeze=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),configurations=configurations,
        primary='Mean response nRMSE on eight new participation requests. Human uses four later heads on their respective profession cohorts; grammar uses next-token to log probability. Source-profile refinement versus identical common support with Euclidean refinement.',
        secondary='Original human source-head error, finer member masks, semantic endpoints, active tangent, reconstruction readout, explanation-specific trained execution on targets2to5, feasibility and resource use.',
        recipe='Frozen source profiles from32source-fit contexts and four full-deletion fractions, mean gradient outer product, trace normalization and.05identity shrinkage. Current-state common2p support,128FISTAsteps. No target response fitting.',
        statistics='2000 paired seed/document/request bootstrap draws. Human documents stratified by profession/gender, paired cohorts across heads. Grammar crossed verb/noun draws with forms paired. Participation and member families resampled separately; endpoints fixed. Two fixed source explanations.',
        evidence='Configuration frozen before these new outcomes. Targets have prior project history, target2 informed development. New contexts, new participation coordinates and new fine masks selected without model outputs.',
        code_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/train_intervention_changes.py','src/ccad/intervention_transport.py','scripts/prepare_intervention_profile_confirmation.py','scripts/analyze_intervention_readouts.py']},
        inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [human_path,grammar_path,request_path,Path(base['source_metric_cache'])]+list(map(Path,configurations))])
    save(OUT/'PROFILE_CONFIRMATION_FREEZE.json',freeze)
    print(json.dumps(dict(human=len(rows),grammar=len(grows),human_requests=len(hq)+len(masks),grammar_requests=len(gq),max_human_tokens=max(map(lambda r:len(r['tokens']),rows)),frozen_at=freeze['written_at_utc'])))


if __name__=='__main__': main()
