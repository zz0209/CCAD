"""Source-only development contexts for whole signed groups and a native reference.

No target codes, target metrics, calibration/audit arrays, or model inference.
Token strata are descriptive samples, not independent observations or labels.
"""
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
import numpy as np
import tokenizers
from run_r011s1_raw_hook_asset import ROOT, entry, aggregate, write_json as write
from ccad.artifacts import sha256, validate_run_directory


def main():
    cfg = dict(run_id='F4_source_contexts_v1_20260906', audit_opened=False,
        queries=[[3,1230],[3,1144],[2,2645],[5,2194]],
        selection='Existing source-effect cases chosen for diverse observed source behavior; no target outcome used in this selection. Historical family is development, not a random population sample.',
        native_reference=[2,2767], split='discovery', rows_per_stratum=5,
        strata=['positive_tail','negative_tail','near_median','same_token_lower_abs_coordinate'],
        original_asset_config='configs/r011f1_euclidean_causal_gate_v1.json',
        parent_config='runs/F4_probability_confirmation_original_v1_20260905/config.resolved.json',
        native_asset_dir='D:/CCAD_Storage/paired_assets/R011_NR1_k32_paired_codes_v1_20260904T060000Z',
        cpu_wall_budget_seconds=120, new_lm_forwards=0)
    run=ROOT/'runs'/cfg['run_id']; run.mkdir(exist_ok=False)
    start=time.perf_counter(); write(run/'config.resolved.json',cfg)
    for n in ['stdout.log','stderr.log','metrics.raw.jsonl']: (run/n).touch()
    code=[]
    for rel in ['scripts/inspect_f4_source_contexts.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py']:
        p=ROOT/rel; dst=run/'source_snapshot'/rel; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.source.contexts.v1',run_id=cfg['run_id'],run_parent='F4',
        purpose='Inspect whole signed source groups and the existing single-atom native reference for falsifiable explanations',
        milestone='C2-C3-interpretability',evidence_level='source_discovery_descriptive_development',
        started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),
        code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,
        mean_constants_source_split='independent mean census for FCC; no centering for native activation reference',
        threshold_source_split='discovery descriptive strata only',statistics_unit='dependent query/document/token contexts',
        device='cpu',seeds=[2,3,5],resource_lease='not_required',
        resource_lease_reason='Bounded sequential source memmaps, four BLAS threads, under 120s budget; no model inference'))
    env=dict(python=platform.python_version(),os=platform.platform(),numpy=np.__version__,tokenizers=tokenizers.__version__,
        torch='not_applicable',transformers='not_imported',cuda='not_applicable',gpu='not_applicable',sae_framework='saved codes only')
    write(run/'environment.json',env); write(run/'status.json',dict(status='RUNNING'))
    inputs=[]; records=[]
    def checked(path):
        path=Path(path); path=path if path.is_absolute() else ROOT/path
        if path not in [Path(i['path']) for i in inputs]:inputs.append(entry(path,'existing CCAD source asset','source discovery or identity','internal; no audit array'))
        return path
    try:
        base=json.loads(checked(cfg['original_asset_config']).read_text())
        old=json.loads(checked(cfg['parent_config']).read_text())
        tm_path=checked(base['token_manifest_path']); tm=json.loads(tm_path.read_text())['outputs']['discovery']
        tok_path=checked(tm_path.parent.parent/tm['path']); assert sha256(tok_path)==tm['sha256']
        tokens=np.fromfile(tok_path,dtype='<u2').reshape(-1,128); flat=tokens.ravel(); n=len(flat)
        seqrows=json.loads(checked(base['sequence_records_path']).read_text())['sequences']
        docs={r['sequence_index']:r['document_ids'] for r in seqrows if r['split']=='discovery'}
        tokenizer=tokenizers.Tokenizer.from_file(str(checked(Path(base['model_local_dir'])/'tokenizer.json')))
        factors=np.load(checked(old['factors_path']),allow_pickle=False)
        lookup={(int(s),int(a),int(t)):i for i,(s,a,t) in enumerate(zip(factors['source_seed'],factors['source_atom'],factors['target_seed']))}
        wanted=set(map(tuple,cfg['queries'])); surfaces={}
        for line in checked(old['surface_path']).read_text().splitlines():
            r=json.loads(line); key=(r['source_seed'],r['source_atom'])
            if key in wanted and r['rank']==1 and r['query_role']=='anchor':
                ids=r['source_candidate_ids']
                if key in surfaces:assert surfaces[key]==ids
                surfaces[key]=ids
        means={s:np.zeros(3072) for s,_ in wanted}
        for line in checked(old['source_census_path']).read_text().splitlines():
            r=json.loads(line)
            if r['seed'] in means:means[r['seed']][r['atom']]=r['mean_code']
        allgroups=[]; sketches=[]
        for s,a in cfg['queries']+[cfg['native_reference']]:
            native=[s,a]==cfg['native_reference']; asset=Path(cfg['native_asset_dir'] if native else base['bulk_asset_dir'])
            checked(asset/'asset_manifest.json'); k=32 if native else 128
            ids=np.asarray([a] if native else surfaces[s,a]); weights=np.ones(1)
            if not native:
                ix=[v for key,v in lookup.items() if key[:2]==(s,a)]
                b=factors['source_basis'][ix[0],:,0].astype(np.float64)
                assert all(np.array_equal(b,factors['source_basis'][i,:,0]) for i in ix)
                dec=np.memmap(checked(asset/'decoders'/f'seed_{s}.float32.bin'),dtype='<f4',mode='r',shape=(3072,768))
                weights=dec[ids].astype(np.float64)@b
            ind=np.memmap(checked(asset/'discovery'/f'seed_{s}'/'top_indices.uint16.bin'),dtype='<u2',mode='r',shape=(n,k))
            acts=np.memmap(checked(asset/'discovery'/f'seed_{s}'/'top_acts.float32.bin'),dtype='<f4',mode='r',shape=(n,k))
            lut=np.full(3072,-1,dtype=int);lut[ids]=np.arange(len(ids)); loc=lut[ind]
            rr,cc=np.nonzero(loc>=0); z=np.zeros((n,len(ids)));np.add.at(z,(rr,loc[rr,cc]),acts[rr,cc])
            center=np.zeros(len(ids)) if native else means[s][ids]
            terms=(z-center)*weights; coord=terms.sum(axis=1)
            valid=np.flatnonzero((flat!=0)&(np.arange(n)%128>0))
            ordered=valid[np.lexsort((valid,-coord[valid]))]
            median=float(np.median(coord[valid])); strata={
                'positive_tail':ordered,'negative_tail':ordered[::-1],
                'near_median':valid[np.lexsort((valid,np.abs(coord[valid]-median)))]}
            chosen=[]; seen=set()
            def pick(order,label):
                seen_docs=set(); count=0
                for p in order:
                    p=int(p); ds=set(docs[p//128])
                    if p in seen or ds&seen_docs:continue
                    chosen.append((p,label));seen.add(p);seen_docs.update(ds);count+=1
                    if count==cfg['rows_per_stratum']:break
            for label,order in strata.items():pick(order,label)
            tailtokens={int(flat[p]) for p,label in chosen if label in ('positive_tail','negative_tail')}
            low=valid[np.isin(flat[valid],list(tailtokens))]
            pick(low[np.lexsort((low,np.abs(coord[low])))],'same_token_lower_abs_coordinate')
            group=f'{s}:{a}'+('/native' if native else '/fcc')
            for p,label in chosen:
                seq,pos=divmod(p,128); seqt=tokens[seq]; boundary=max(np.flatnonzero(seqt[:pos]==0),default=-1)
                begin=max(boundary+1,pos-40); end=min(128,pos+1)
                tt=terms[p]; ao=np.argsort(-np.abs(tt),kind='stable'); denom=float(np.abs(tt).sum())
                row=dict(run_id=cfg['run_id'],metric_version='v1',group=group,source_seed=s,source_atom=a,
                    stratum=label,sequence=seq,position=pos,document_ids=docs[seq],coordinate=float(coord[p]),
                    token_id=int(seqt[pos]),token=tokenizer.decode([int(seqt[pos])]),
                    prefix=tokenizer.decode([int(v) for v in seqt[begin:pos]]),
                    observed_next_token=tokenizer.decode([int(v) for v in seqt[end:end+1]]),
                    anchor_abs_term_share=float(abs(tt[list(ids).index(a)])/denom) if denom else None,
                    signed_terms=[dict(atom=int(ids[j]),code=float(z[p,j]),mean=float(center[j]),weight=float(weights[j]),term=float(tt[j])) for j in ao],
                    context_has_future=False,observed_next_token_is_separate=True)
                records.append(row)
            top=ordered[:1000]; vals,counts=np.unique(flat[top],return_counts=True); tidx=np.lexsort((vals,-counts))[:12]
            allgroups.append(dict(group=group,members=ids.tolist(),weights=weights.tolist(),mean=center.tolist(),
                eligible_tokens=len(valid),positive_nonzero_tokens=int(np.count_nonzero(z[:,list(ids).index(a)])),
                coordinate_quantiles=np.quantile(coord[valid],[0,.01,.1,.5,.9,.99,1]).tolist(),
                top1000_tokens=[dict(id=int(vals[j]),token=tokenizer.decode([int(vals[j])]),count=int(counts[j])) for j in tidx],
                sampled_rows=sum(r['group']==group for r in records)))
            sketches.append(f'## {group}\n'+ '\n'.join(f"- {r['stratum']} c={r['coordinate']:.5g} anchor={r['anchor_abs_term_share']}: {r['prefix']!r} **{r['token']!r}** -> {r['observed_next_token']!r}" for r in records if r['group']==group))
            if time.perf_counter()-start>cfg['cpu_wall_budget_seconds']:raise TimeoutError('Source context budget exceeded')
        factors.close()
        (run/'metrics.raw.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        write(run/'inputs.json',dict(inputs=inputs)); write(run/'groups.json',dict(groups=allgroups))
        elapsed=time.perf_counter()-start
        write(run/'metrics.summary.json',dict(groups=allgroups,rows=len(records),wall_seconds=elapsed,new_lm_forwards=0,
            metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/inspect_f4_source_contexts.py',
            generator_script_sha256=code[0]['sha256'],scope='Discovery only, descriptive case development, no semantic labels or target tests',
            checks=dict(rows_per_group=all(g['sampled_rows']==20 for g in allgroups),all_signed_sums=all(abs(sum(t['term'] for t in r['signed_terms'])-r['coordinate'])<1e-10 for r in records))))
        (run/'SOURCE_CONTEXTS.md').write_text('# Source discovery context inspection\n\nWhole centered signed FCC groups; native reference uses uncentered activation. All labels are sample strata. Prefix excludes future; observed next token shown separately. Samples within/across strata can share documents and are not independent.\n\n'+'\n\n'.join(sketches)+'\n',encoding='utf-8')
        write(run/'status.json',dict(status='PASS',wall_seconds=elapsed))
        validation=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=validation.ok,errors=validation.errors))
        if not validation.ok:raise ValueError(validation.errors)
        print(json.dumps(dict(run_id=cfg['run_id'],rows=len(records),wall_seconds=elapsed,contract=validation.ok)))
    except Exception:
        (run/'stderr.log').write_text(traceback.format_exc());write(run/'inputs.json',dict(inputs=inputs))
        write(run/'status.json',dict(status='FAIL',wall_seconds=time.perf_counter()-start));raise


if __name__=='__main__':main()
