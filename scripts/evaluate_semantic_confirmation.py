"""Execute a frozen source-family and native-writer suite on new cities.

No fitting, source selection or target semantic supervision occurs here.
Old held rows are replay witnesses; new cities retain a separate split.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
from run_causalgym_multisite import MultisiteWork,ROOT,write
from run_ravel_semantic_source import expand_delta
from run_ravel_source_coverage import semantic_measure


class ConfirmationWork(MultisiteWork):
    def batches(self, ids):
        # The old panel ends with a short batch. Appending new cities must not
        # change that batch's model or encoder GEMM shape before source replay.
        boundary=getattr(self,'preserved_prefix_rows',None)
        if boundary is not None and len(ids)==len(self.panel) and np.array_equal(ids,np.arange(len(self.panel))):
            yield from super().batches(ids[:boundary])
            yield from super().batches(ids[boundary:])
        else:
            yield from super().batches(ids)


def source_bases(w,spec):
    bases=w.torch.as_tensor(np.load(w.checked(ROOT/spec['basis_path']))['bases'],device=w.device)
    if spec['family']=='decoded_independent':
        # The original row-orthogonal parametrization returns each [rank,hook]
        # matrix with column-major stride (1,rank). Stacking/exporting its
        # values makes it row-major and selects different float32 GEMMs.
        # Retain the original arithmetic as well as the exact saved values.
        bases=bases.transpose(-1,-2).contiguous().transpose(-1,-2)
    return bases


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);path=ap.parse_args().config;cfg=json.loads(path.read_text())
    files=['scripts/evaluate_semantic_confirmation.py','scripts/run_ravel_semantic_source.py','scripts/run_ravel_source_coverage.py',
        'scripts/run_causalgym_multisite.py','scripts/run_causalgym_native_participation.py','scripts/run_causalgym_native_transfer.py',
        'scripts/run_r011s1_raw_hook_asset.py','src/ccad/semantic_family.py','src/ccad/semantic_readout.py','src/ccad/ravel_controls.py',
        'src/ccad/semantic_participation.py','src/ccad/native_operation.py','src/ccad/activation_contract.py','src/ccad/artifacts.py']
    w=ConfirmationWork(cfg,path,files);error=None
    try:
        from safetensors import safe_open
        from ccad.semantic_family import apply_torch
        from ccad.native_operation import adaptive_writable_support,batched_project_native
        freeze_path=ROOT/cfg['freeze_path'];freeze=json.loads(w.checked(freeze_path).read_text())
        if hashlib.sha256(freeze_path.read_bytes()).hexdigest()!=cfg['freeze_sha256']:raise ValueError('Freeze identity differs')
        for key,value in freeze['execution'].items():
            if cfg.get(key)!=value:raise ValueError('Frozen execution setting differs: '+key)
        for record in freeze['files']:
            f=ROOT/record['path']
            if hashlib.sha256(w.checked(f).read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Frozen input changed: '+str(f))
        for record in freeze['sae_inputs']:
            f=w.checked(Path(record['path']))
            with f.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
            if actual!=record['sha256']:raise ValueError('Frozen SAE material changed')
        prepared=json.loads(Path(cfg['prepared_panel']).read_text());old=json.loads((ROOT/freeze['old_panel']).read_text())
        if prepared['rows'][:len(old['rows'])]!=old['rows']:raise ValueError('Old source panel prefix differs')
        w.preserved_prefix_rows=len(old['rows'])
        w.setup();torch=w.torch;ids=np.arange(len(w.panel));fresh=ids[[r['split']=='confirmation' for r in w.panel]]
        held=ids[[r['split']=='held_component_development' for r in w.panel]];all_eval=np.concatenate([held,fresh])
        if len(fresh)!=freeze['confirmation_rows']:raise ValueError('Unexpected confirmation sample size')
        valid,align=w.alignment(ids,cfg['mode'])
        if not valid.all() or any(len(a)!=1 for a in align):raise ValueError('Single-site alignment required')
        w.semantic_positions=np.asarray([a[0][0] for a in align]);donorpos=np.asarray([a[0][1] for a in align])
        if not np.array_equal(w.semantic_positions[w.donors],donorpos):raise ValueError('Donor token alignment differs')
        material={};new_slice=slice(len(held),None);old_slice=slice(0,len(held))
        old_ids=np.arange(len(old['rows']))
        for seed in cfg['seeds']:
            s=w.load_sae(seed,w.semantic_positions)
            d=torch.as_tensor(s['decoder'],device=w.device);z=torch.as_tensor(s['codes'][all_eval],device=w.device)
            dz=torch.as_tensor(s['codes'][w.donors[all_eval]]-s['codes'][all_eval],device=w.device)
            with torch.no_grad():
                decoded=dz@d
                # Preserve the producer's full old-panel decoder GEMM before
                # selecting held rows; the new-city projection is separate.
                old_dz=torch.as_tensor(s['codes'][w.donors[old_ids]]-s['codes'][old_ids],device=w.device)
                old_decoded=(old_dz@d)[held]
            material[seed]=dict(decoder=d,base=z,difference=dz,decoded=decoded,old_decoded=old_decoded,cfg=s['cfg']);del s,old_dz
        hbase=torch.as_tensor(w.hidden[fresh,w.semantic_positions[fresh]],device=w.device)
        raw_difference=torch.as_tensor(w.hidden[w.donors[all_eval],w.semantic_positions[w.donors[all_eval]]]-w.hidden[all_eval,w.semantic_positions[all_eval]],device=w.device)
        raw_bases=torch.as_tensor(np.load(w.checked(ROOT/freeze['raw_basis']))['bases'],device=w.device)
        raw_singleton_bases=torch.as_tensor(np.load(w.checked(ROOT/freeze['raw_singleton_basis']))['bases'],device=w.device)
        controls=cfg['binary_controls']+cfg['fractional_controls'];diagnostics=[];witnesses=[]
        name=lambda c:''.join(str(int(v)) if v in [0,1] else str(v) for v in c)
        fixed=min(fresh,key=lambda i:(w.panel[i]['component'],i));fixed_local=int(np.flatnonzero(fresh==fixed)[0])
        # Replay every old source before the first new-city intervention.
        for source_spec in freeze['sources']:
            source=source_spec['source_seed'];ms=material[source]
            bases=source_bases(w,source_spec)
            saved=np.load(w.checked(ROOT/source_spec['replay_outputs']))
            if not np.array_equal(saved['row_ids'],held):raise ValueError('Old held row identity differs')
            for c in cfg['binary_controls']:
                q=apply_torch(ms['old_decoded'],bases,c)
                delta=expand_delta(w,held,q).cpu().numpy();lp=np.empty_like(saved[name(c)])
                for local in w.batches(np.arange(len(held))):lp[local],_=w.forward(held[local],delta[local])
                discrepancy=float(np.max(np.abs(lp-saved[name(c)])));witnesses.append(dict(source_seed=source,control=c,max_logprob_error=discrepancy))
                write(w.run/'source_replay_witness.json',dict(rows=witnesses,max_logprob_error=max(r['max_logprob_error'] for r in witnesses),tolerance=cfg['replay_logprob_tolerance']))
                if discrepancy>=cfg['replay_logprob_tolerance']:raise ValueError(f'Frozen source replay failed: source={source}, control={c}, max_logprob_error={discrepancy}')
            del bases,saved,lp,delta,q
        w.progress('ALL_FROZEN_SOURCES_REPLAYED',source_count=len(freeze['sources']),max_logprob_error=max(r['max_logprob_error'] for r in witnesses))
        for source_spec in freeze['sources']:
            source=source_spec['source_seed'];target=source_spec['target_seed'];ms=material[source];mt=material[target]
            bases=source_bases(w,source_spec)
            atom=np.load(w.checked(ROOT/source_spec['atom_map']));matched=torch.as_tensor(atom['target_indices'].astype(np.int64),device=w.device);beta=torch.as_tensor(atom['beta'],device=w.device)
            with torch.no_grad():atom_prediction=(mt['difference'][new_slice][:,matched]*beta)@ms['decoder']
            sae_path=Path(cfg['sae_root'])/f'seed_{target}/sae.safetensors'
            with safe_open(w.checked(sae_path),framework='pt',device=str(w.device)) as f:ew,eb,db=[f.get_tensor(k) for k in ['encoder.weight','encoder.bias','b_dec']]
            z=mt['base'][new_slice];d=mt['decoder'];references={}
            def evaluate(q,method,c,reference=None,**extra):
                return semantic_measure(w,fresh,expand_delta(w,fresh,q).cpu().numpy(),method,cfg['mode'],c,
                    seed=source,target_seed=target,reference=reference,control_is_binary=all(v in [0,1] for v in c),
                    source_family=source_spec['family'],confirmation_freeze_sha256=cfg['freeze_sha256'],
                    primary_fractional_endpoint='KL to identical frozen source control; categorical outcomes descriptive only',**extra)
            for ci,c in enumerate(controls):
                key=name(c)
                source_q=apply_torch(ms['decoded'][new_slice],bases,c);reference=evaluate(source_q,'frozen_source',c)
                np.savez_compressed(w.run/f's{source}_{key}_source_outputs.npz',row_ids=fresh,logprobs=reference.astype(np.float32))
                evaluate(torch.zeros_like(source_q),'unedited',c,reference)
                evaluate(apply_torch(raw_difference[new_slice],raw_bases,c),'raw_supervised_family',c,reference)
                evaluate(apply_torch(raw_difference[new_slice],raw_singleton_bases,c),'raw_supervised_singleton',c,reference)
                evaluate(apply_torch(raw_difference[new_slice],bases,c),'raw_shared_space',c,reference)
                evaluate(apply_torch(atom_prediction,bases,c),'atom_pw_mcc',c,reference)
                q=apply_torch(mt['decoded'][new_slice],bases,c);evaluate(q,'direct_shared_space',c,reference)
                if c in cfg['binary_controls'][:3]:
                    wrong=apply_torch(mt['decoded'][new_slice],bases,np.roll(c,1))
                    wrong=wrong*(torch.linalg.vector_norm(q,dim=1)/torch.linalg.vector_norm(wrong,dim=1).clamp_min(1e-12))[:,None]
                    evaluate(wrong,'wrong_attribute_norm_matched',c,reference)
                with torch.no_grad():
                    act,ix,_=w.encode_kernel(hbase+q-db,ew,eb,mt['cfg']['k'],mt['cfg']['activation'])
                    edited=torch.zeros_like(z).scatter_(1,ix,act);full_u=edited-z;realized=full_u@d
                evaluate(realized,'target_reencode',c,reference)
                np.savez_compressed(w.run/f's{source}_t{target}_{key}_reencode.npz',row_ids=fresh,code_increment=full_u.cpu().numpy())
                diagnostics.append(dict(source_seed=source,target_seed=target,control=list(c),method='target_reencode',
                    changed_members=(full_u!=0).sum(1).cpu().tolist(),minimum_final_state=float(edited.min()),
                    squared_writer_error=((realized-q)**2).sum(1).cpu().tolist(),desired_energy=q.square().sum(1).cpu().tolist()))
                start=time.perf_counter();selected=adaptive_writable_support(q.cpu().numpy(),z.cpu().numpy(),d.cpu().numpy(),max(cfg['writer_budgets']),np.ones(len(d),bool),device=str(w.device))
                torch.cuda.synchronize();selection_time=time.perf_counter()-start
                supports=[(f'adaptive_native{b}',selected[:,:b],None) for b in cfg['writer_budgets']]
                # A separate prefix matches the actual re-encoding edit count
                # in each context. Padded columns have zero decoder and state.
                matched_counts=(full_u!=0).sum(1)
                if int(matched_counts.max())>selected.shape[1]:raise ValueError('Greedy path cannot match reencoding budget')
                matched_width=max(1,int(matched_counts.max()))
                matched_mask=torch.arange(matched_width,device=w.device)[None,:]<matched_counts[:,None]
                supports.append(('adaptive_native_reencode_count',selected[:,:matched_width],matched_mask))
                rng=np.random.default_rng(cfg['random_support_seed']+1000*source+100*target+ci);random=[]
                for row in z.cpu().numpy():
                    active=np.flatnonzero(row>0);inactive=np.flatnonzero(row==0)
                    random.append(np.concatenate([active,rng.choice(inactive,cfg['random_budget']-len(active),replace=False)]))
                supports.append((f'random_base_native{cfg["random_budget"]}',np.asarray(random),None))
                for method,support,support_mask in supports:
                    begin=time.perf_counter();index=torch.as_tensor(support,device=w.device);local_z=z.gather(1,index);local_d=d[index]
                    if support_mask is not None:
                        local_z=local_z*support_mask;local_d=local_d*support_mask[:,:,None]
                    member_counts=(support_mask.sum(1).cpu().numpy() if support_mask is not None else np.full(len(fresh),support.shape[1]))
                    with torch.no_grad():u,realized,diag=batched_project_native(q,local_z,local_d,max_steps=cfg['writer_steps'],tolerance=cfg['writer_tolerance'])
                    torch.cuda.synchronize();solve_time=time.perf_counter()-begin
                    if diag['minimum_final_state']< -1e-7:raise ValueError('Infeasible native code')
                    evaluate(realized,method,c,reference,candidate_members=support.shape[1],
                             candidate_count_scope='Padded maximum; actual per-row allowed count is saved in the support artifact and diagnostics.' if support_mask is not None else 'Same candidate count for all rows')
                    np.savez_compressed(w.run/f's{source}_t{target}_{key}_{method}.npz',row_ids=fresh,members=support,
                        active_support_mask=np.ones(support.shape,bool) if support_mask is None else support_mask.cpu().numpy(),
                        code_increment=u.cpu().numpy(),base_selected_codes=local_z.cpu().numpy())
                    diagnostics.append(dict(source_seed=source,target_seed=target,control=list(c),method=method,candidate_members=member_counts.tolist(),
                        changed_members=(u!=0).sum(1).cpu().tolist(),desired_energy=q.square().sum(1).cpu().tolist(),
                        selection_seconds=selection_time if method.startswith('adaptive') else None,
                        selection_timing_scope='One shared maximum-size path per input/control; repeated across prefixes, do not sum.',solve_seconds=solve_time,**diag))
                    if method=='adaptive_native256' and c in cfg['binary_controls'][:3]:
                        example=w.run/f'example_s{source}_t{target}_{key}.npz'
                        np.savez_compressed(example,bases=bases.cpu().numpy(),decoded_difference=mt['decoded'][new_slice][fixed_local:fixed_local+1].cpu().numpy(),
                            members=support[fixed_local],decoder=local_d[fixed_local].cpu().numpy(),base_codes=local_z[fixed_local:fixed_local+1].cpu().numpy(),
                            control=np.asarray(c),expected_readout=q[fixed_local:fixed_local+1].cpu().numpy(),
                            expected_native_delta=realized[fixed_local:fixed_local+1].cpu().numpy(),expected_code_increment=u[fixed_local:fixed_local+1].cpu().numpy())
                    del index,local_z,local_d,u
                write(w.run/'native_writer_diagnostics.json',dict(rows=diagnostics,scope=cfg['scope']))
                w.progress('FROZEN_CITY_CONTROL',source_seed=source,target_seed=target,control=c,new_rows=len(fresh),selection_seconds=selection_time,last_solve_seconds=solve_time)
                del reference,full_u,edited,realized
            del bases,atom_prediction,ew,eb,db
        write(w.run/'fixed_example.json',dict(row=w.panel[fixed],selection='First component/row by stored identity before outcomes; examples are not chosen for success.'))
        w.checks['frozen_source_replayed']=all(r['max_logprob_error']<cfg['replay_logprob_tolerance'] for r in witnesses)
        w.checks['confirmation_city_disjoint']=not {r['entity'] for r in w.panel if r['split']=='confirmation'}&{r['entity'] for r in w.panel if r['split']!='confirmation'}
        w.checks['five_cyclic_sources']=len(freeze['sources'])==5 and {r['source_seed'] for r in freeze['sources']}==set(range(1,6))
    except Exception:error=traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
