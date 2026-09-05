"""Source-only donor-class control on frozen case recipients; no endpoint reads."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from run_f4_source_reference_causal import ROOT,np,sha256,write,jsonl,validate_run_directory
from f4_case_details import token_class,ordered_class_donor
from inspect_f4_atom_participation import participation
from summarize_f4_source_scope import selected


def prepare(panel,inputs):
    paths={k:ROOT/panel[k+'_path'] for k in ('reference_config','selection','case_selection')}
    expected={k:panel[k+'_sha256'] for k in paths}
    cfg=json.loads(paths['reference_config'].read_text());cases=json.loads(paths['case_selection'].read_text())
    for key in ('surface','factors','token_manifest'):
        paths[key]=ROOT/cfg[key+'_path'];expected[key]=cfg[key+'_sha256']
    paths['asset_manifest']=Path(cfg['bulk_asset_dir'])/'asset_manifest.json';expected['asset_manifest']=cfg['asset_manifest_sha256']
    paths['raw_manifest']=Path(cfg['raw_hook_asset_dir'])/'raw_hook_manifest.json';expected['raw_manifest']=cfg['raw_hook_manifest_sha256']
    paths['scope_rule']=ROOT/cfg['source_scope']['path'];expected['scope_rule']=cfg['source_scope']['sha256']
    for k,p in paths.items():
        if sha256(p)!=expected[k]:raise ValueError('Input changed: '+k)
        inputs.append(dict(path=str(p.resolve()),sha256=sha256(p),bytes=p.stat().st_size,source=panel['id'],license_or_access_boundary='internal existing data; no audit arrays',role=k))
    rule=json.loads(paths['scope_rule'].read_text())['rule']
    units=json.loads(paths['selection'].read_text())['queries']
    surface={(r['source_seed'],r['source_atom'],r['target_seed']):r for r in jsonl(paths['surface']) if r['rank']==1 and r['query_role']=='anchor'}
    factors=np.load(paths['factors'],allow_pickle=False)
    index={(int(s),int(a),int(t)):i for i,(s,a,t) in enumerate(zip(factors['source_seed'],factors['source_atom'],factors['target_seed']))}
    manifest=json.loads(paths['asset_manifest'].read_text());nt=next(r['tokens'] for r in manifest['splits'] if r['split']=='calibration')
    length=cfg['context_length'];h=cfg['hook_hidden_size'];asset=Path(cfg['bulk_asset_dir'])
    rawspec=next(r for r in json.loads(paths['raw_manifest'].read_text())['splits'] if r['split']=='calibration')
    raw=np.memmap(rawspec['path'],dtype='<f4',mode='r',shape=tuple(rawspec['shape']))
    tokenmeta=json.loads(paths['token_manifest'].read_text())['outputs']['calibration']
    tokenpath=ROOT/'runs'/cfg['paired_corpus_run']/tokenmeta['path']
    if sha256(tokenpath)!=tokenmeta['sha256']:raise ValueError('Token bytes changed')
    inputs.append(dict(path=str(tokenpath),sha256=sha256(tokenpath),bytes=tokenpath.stat().st_size,source=panel['id'],license_or_access_boundary='existing corpus tokens',role='token_array'))
    tokens=np.memmap(tokenpath,dtype='<u2',mode='r').reshape(-1,length)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import transformers
    tokenizer=transformers.AutoTokenizer.from_pretrained(cfg['model_local_dir'],local_files_only=True)
    output=[]
    for unit in units:
        s,a=unit['source_seed'],unit['source_atom'];ids=surface[s,a,unit['targets'][0]]['source_candidate_ids']
        b=factors['source_basis'][index[s,a,unit['targets'][0]],:,:1].astype(np.float64)
        decoder=np.memmap(asset/'decoders'/f'seed_{s}.float32.bin',dtype='<f4',mode='r',shape=(cfg['num_latents'],h))
        weights=decoder[ids].astype(np.float64)@b
        ix=np.memmap(asset/'calibration'/f'seed_{s}'/'top_indices.uint16.bin',dtype='<u2',mode='r',shape=(nt,cfg['k']))
        act=np.memmap(asset/'calibration'/f'seed_{s}'/'top_acts.float32.bin',dtype='<f4',mode='r',shape=(nt,cfg['k']))
        z={};classes={};texts={}
        for e in unit['sequences']:
            seq=e['sequence'];sl=slice(seq*length,(seq+1)*length);dense=np.zeros((length,cfg['num_latents']))
            np.add.at(dense,(np.arange(length)[:,None],ix[sl]),act[sl]);z[seq]=dense[:,ids]
            for p in e['intervention_positions']:
                text=tokenizer.decode([int(tokens[seq,p])]);texts[seq,p]=text;classes[seq,p]=token_class(text)
        coordinates={seq:value@weights for seq,value in z.items()}
        for original in [c for c in cases['choices'] if c['source_seed']==s and c['source_atom']==a]:
            entry=original['entry'];record=dict(source_seed=s,source_atom=a,condition=original['condition'],original_entry=entry,original_source_scope=original['source_scope'])
            if entry is None:
                output.append(dict(record,entry=None,source_scope=None,matching_status='ORIGINAL_NO_SUPPORT'));continue
            if sum(e==entry for e in unit['sequences'])!=1:raise ValueError('Frozen recipient changed')
            seq=entry['sequence'];pos=entry['intervention_positions'];donors=[e for e in unit['sequences'] if e['condition']!=entry['condition']]
            choice,tested=ordered_class_donor(seq,pos,donors,coordinates,classes)
            record.update(candidate_donors=tested,recipient_classes=[classes[seq,p] for p in pos],recipient_tokens=[texts[seq,p] for p in pos])
            if choice is None:
                output.append(dict(record,entry=None,source_scope=None,matching_status='NO_CLASS_COMPATIBLE_DONOR'));continue
            energy,_,dp,donor=choice
            if set(entry['document_ids'])&set(donor['document_ids']):raise ValueError('Overlapping donor document')
            matched=dict(entry,donor_sequence=donor['sequence'],donor_positions=dp,donor_document_ids=donor['document_ids'],donor_source_difference_energy=energy,donor_status='SELECTED_SOURCE_ONLY')
            zd=z[seq][pos]-z[donor['sequence']][dp]
            stat=participation(zd,np.zeros(len(ids)),weights)
            hooks=raw[np.asarray(pos,dtype=int)+seq*length].astype(np.float64);norm=float(np.linalg.norm(hooks));fraction=float(np.sqrt(stat['aggregate_energy'])/norm) if norm else None
            scope=dict(source_seed=s,source_atom=a,rank=1,condition=entry['condition'],sequence=seq,**stat,natural_source_hook_fraction=fraction,supported=bool(stat['aggregate_energy']>0))
            scope['selected']=selected(scope,rule)
            scale=min(1.,cfg['maximum_source_hook_fraction']/fraction) if fraction else 1.
            coordinate=(zd@weights)[:,0];hn=np.linalg.norm(hooks,axis=1);delta_norm=np.abs(coordinate)*np.linalg.norm(b)
            positional=[dict(position=p,donor_position=d,recipient_token=texts[seq,p],donor_token=texts[donor['sequence'],d],token_class=classes[seq,p],source_coordinate=float(c),recipient_hook_norm=float(n),natural_hook_fraction=float(v/n) if n else None,dosed_hook_fraction=float(v*scale/n) if n else None) for p,d,c,n,v in zip(pos,dp,coordinate,hn,delta_norm)]
            unchanged=all(matched[k]==entry[k] for k in ('donor_sequence','donor_positions','donor_document_ids'))
            record.update(entry=matched if scope['supported'] else None,source_scope=scope,positions=positional,source_dose_scale=scale,
                          matching_status=('UNCHANGED_REUSABLE' if unchanged else 'CHANGED_SUPPORTED') if scope['supported'] else 'NO_SOURCE_DIFFERENCE')
            output.append(record)
    exposed=panel.get('prior_endpoint_exposure',True)
    return dict(panel=panel['id'],donor_override=True,evaluate_changed_only=exposed,choices=output,
        rule='Freeze recipient and original allowed positions. For each opposite-condition donor use its first n sorted original positions, as before; retain only exact coarse token-class sequence matches, choose largest source difference energy with smallest donor sequence tie. No permutation, extra position pool, target coordinates or endpoints.',
        prior_endpoint_exposure=exposed,scope='Exposed-document development, coarse character class not POS/semantics. Unchanged pairs reuse previous raw outcomes.' if exposed else 'Fresh-document frozen source workflow. Coarse character class not POS/semantics. Matching status unchanged refers only to original donor, not reused outcomes; all selected pairs require new forwards.',tokenizer_revision=cfg['model_revision'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False)
    start=time.perf_counter();now=datetime.now(timezone.utc).isoformat();write(run/'config.resolved.json',cfg)
    sources=[Path(__file__)]+[ROOT/'scripts'/n for n in ('f4_case_details.py','run_f4_source_reference_causal.py','inspect_f4_atom_participation.py','summarize_f4_source_scope.py')]+[ROOT/'src/ccad'/n for n in ('artifacts.py','activation_contract.py')]
    code=[]
    for p in sorted(sources,key=lambda p:p.relative_to(ROOT).as_posix()):
        rel=p.relative_to(ROOT).as_posix();dest=run/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path=dest.relative_to(run).as_posix()))
    ch=hashlib.sha256(''.join(f"{r['path']}:{r['sha256']}\n" for r in code).encode()).hexdigest()
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=ch,snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.donor.class.preparation.v1',run_id=cfg['run_id'],run_parent='F4',purpose='Source-only matched-donor control preparation',milestone='M4',evidence_level='exposed_document_source_preparation',started_utc=now,project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=ch,source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='mean cancels in difference',threshold_source_split='frozen source applicability; current coarse class control',statistics_unit='fixed recipient/query/seed dependent',device='cpu',seeds=[1,2,3,4,5],resource_lease='not_required_bounded_CPU',resource_lease_reason='Small source rows and tokenizer only; no model forward, fit, full-array read, or long CPU work'))
    write(run/'status.json',dict(status='RUNNING',updated_utc=now));inputs=[];records=[];error=None
    try:
        for panel in cfg['panels']:
            payload=prepare(panel,inputs);write(run/(panel['id']+'_matching.json'),payload)
            records.extend(dict(panel=panel['id'],**r) for r in payload['choices'])
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';(run/'stderr.log').write_text(traceback.format_exc())
    write(run/'inputs.json',dict(inputs=inputs));(run/'metrics.raw.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    counts={p:{status:sum(r['panel']==p and r['matching_status']==status for r in records) for status in sorted({r['matching_status'] for r in records})} for p in ('original','expanded')}
    checks=dict(expected_conditions=len(records)==32,source_only_no_endpoint_arrays=True)
    status='PASS' if error is None and all(checks.values()) else 'FAIL'
    write(run/'metrics.summary.json',dict(status=status,error=error,checks=checks,counts=counts,wall_seconds=time.perf_counter()-start,model_forwards=0,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/prepare_f4_token_class_donors.py',generator_script_sha256=sha256(Path(__file__))))
    import transformers
    write(run/'environment.json',dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,transformers=transformers.__version__,platform=platform.platform(),device='cpu',cuda='not_used',sae='existing source rows',torch='tokenizer imports only, no tensors/forwards'))
    write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()));write(run/'stdout.log',dict(status=status,error=error,counts=counts))
    if not (run/'stderr.log').exists():(run/'stderr.log').write_text('')
    contract=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=contract.ok,errors=list(contract.errors)))
    print(json.dumps(dict(status=status,error=error,counts=counts,contract_ok=contract.ok,contract_errors=list(contract.errors))))
    return 0 if status=='PASS' and contract.ok else 1


if __name__=='__main__':raise SystemExit(main())
