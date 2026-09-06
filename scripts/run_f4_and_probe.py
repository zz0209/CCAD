"""Single-position source group swaps for frozen authored and/or contrasts."""
import argparse,json,os,platform,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np
from run_r011s1_raw_hook_asset import ROOT,entry,aggregate,write_json as write
from ccad.artifacts import sha256,validate_run_directory
from ccad.activation_contract import HookPointContract,extract_primary_hook_tensor,replace_primary_hook_tensor
from f4_interpretation_endpoint import contrast,masses


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text());run=ROOT/'runs'/cfg['run_id'];run.mkdir(exist_ok=False);start=time.perf_counter()
    write(run/'config.resolved.json',cfg)
    for n in ['stdout.log','stderr.log','metrics.raw.jsonl']:(run/n).touch()
    code=[]
    for rel in ['scripts/run_f4_and_probe.py','scripts/f4_interpretation_endpoint.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py','src/ccad/activation_contract.py']:
        p=ROOT/rel;dst=run/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes());code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write(run/'manifest.json',dict(schema_version='fcc.and.source.probe.v1',run_id=cfg['run_id'],run_parent='F4',purpose='Source group explanation through fixed one-position and/or swaps',milestone='C2-C3-interpretability',
        evidence_level=cfg.get('evidence_level','authored_source_feasibility_development'),started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),
        code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='independent mean; cancels in matched donor difference',
        threshold_source_split='fixed config before new forwards',statistics_unit=cfg.get('statistics_unit','eight authored pairs, two families, four shared topics; dependent cases'),device=cfg['device'],seeds=[cfg['source_seed']],
        resource_lease='gpu-0 resource_manager.run',resource_lease_reason='64 bounded forwards; no fitting, heavy CPU or disk work'))
    write(run/'status.json',dict(status='RUNNING'));inputs=[];rows=[];arrays={};forwards=0
    def checked(path,expected=None):
        p=Path(path);p=p if p.is_absolute() else ROOT/p;e=entry(p,'existing frozen source asset','input','internal/Apache-2.0 model; no audit')
        if expected and e['sha256']!=expected:raise ValueError(f'Input hash differs: {p}')
        inputs.append(e);return p
    try:
        lease=json.loads(subprocess.check_output([sys.executable,str(ROOT.parent/'.resource_manager/resource_manager.py'),'status','--resource','gpu-0'],text=True));write(run/'resource_status_at_start.json',lease)
        old=json.loads(checked(cfg['asset_config']).read_text());spec=next(s for s in old['saes'] if s['seed']==cfg['source_seed']);sp=ROOT/spec['path']
        checked(sp/'sae.safetensors',spec['sha256']);checked(sp/'cfg.json');checked(Path(old['model_local_dir'])/'config.json');checked(Path(old['model_local_dir'])/'tokenizer.json');checked('.aris/compute/local-r006b1-env-spec.json')
        g=next(g for g in json.loads(checked(cfg['groups_path'],cfg['groups_sha256']).read_text())['groups'] if g['group']==f"{cfg['source_seed']}:{cfg['source_atom']}/fcc")
        f=np.load(checked(cfg['factors_path'],cfg['factors_sha256']),allow_pickle=False)
        ix=np.flatnonzero((f['source_seed']==cfg['source_seed'])&(f['source_atom']==cfg['source_atom']));b=f['source_basis'][ix[0],:,0].astype(np.float64)
        if not all(np.array_equal(b,f['source_basis'][i,:,0]) for i in ix):raise ValueError('Source basis depends on target')
        f.close();ids=np.array(g['members']);w=np.array(g['weights']);mean=np.array(g['mean'])
        rng=np.random.default_rng(cfg['random_seed']);random=rng.normal(size=len(b));random-=b*np.dot(b,random)/np.dot(b,b);random/=np.linalg.norm(random)
        import torch,transformers
        from sparsify import SparseCoder
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
        model=transformers.AutoModelForCausalLM.from_pretrained(old['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(cfg['device']);model.config.use_cache=False
        tokenizer=transformers.AutoTokenizer.from_pretrained(old['model_local_dir'],local_files_only=True);sae=SparseCoder.load_from_disk(sp,device=cfg['device']).eval()
        decoder=sae.W_dec.detach().cpu().numpy().astype(np.float64);weight_error=float(np.max(abs(decoder[ids]@b-w)))
        if weight_error>1e-10:raise ValueError('Saved source coefficients do not replay decoder/basis')
        write(run/'environment.json',dict(python=platform.python_version(),os=platform.platform(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),sae_framework='sparsify '+old['sparsify_commit']))
        contract=HookPointContract(old['hook_module_path'],5,'resid_post',768);module=model.get_submodule(old['hook_module_path'])
        def single(words):
            ts=[tokenizer.encode(x,add_special_tokens=False) for x in words]
            if any(len(t)!=1 for t in ts):raise ValueError(f'Contrast is not single-token: {words}')
            return [t[0] for t in ts]
        clause=single(cfg['clause_start_tokens']);prepared=[]
        if 'prepared_inputs_path' in cfg:
            payload=json.loads(checked(cfg['prepared_inputs_path'],cfg['prepared_inputs_sha256']).read_text())
            prepared=payload['cases'];items=payload['item_token_ids']
            if payload['clause_token_ids']!=clause or cfg['endpoint']!='pronoun_logodds':raise ValueError('Frozen endpoint differs')
            if not all(x is None for x in items):raise ValueError('Natural endpoint must use fixed-set complement')
            if not prepared or len(prepared)%2 or len(prepared)>16:raise ValueError('Invalid natural case budget')
        else:
            items=[single(p['items']) for p in cfg['pairs']]
            for pair_index,p in enumerate(cfg['pairs']):
                for conj in cfg['conjunctions']:
                    text=cfg['common_prefix']+p['prefix']+conj;ts=tokenizer.encode(text,add_special_tokens=False)
                    prepared.append(dict(pair_index=pair_index,family=p['family'],topic=p['topic'],conjunction=conj,text=text,token_ids=ts))
        for i in range(0,len(prepared),2):
            if prepared[i]['token_ids'][:-1]!=prepared[i+1]['token_ids'][:-1]:raise ValueError('Matched pair changes more than final token')
        write(run/'authored_inputs.json',dict(cases=prepared,clause_token_ids=clause,item_token_ids=items));write(run/'inputs.json',dict(inputs=inputs));numeric=time.perf_counter()
        def forward(ts,delta=None):
            nonlocal forwards
            if forwards>=cfg['maximum_forwards']:raise ValueError('Forward budget exceeded')
            hidden=[]
            def hook(m,i,out):
                h=extract_primary_hook_tensor(out,contract);hidden.append(h[0,-1].detach().cpu().numpy().copy())
                if delta is None:return out
                edit=h.clone();edit[0,-1]+=torch.tensor(delta,device=h.device,dtype=h.dtype);return replace_primary_hook_tensor(out,edit,contract)
            handle=module.register_forward_hook(hook)
            try:
                with torch.no_grad():lg=model(torch.tensor([ts],device=cfg['device'])).logits[0,-1].cpu().numpy().astype(np.float64)
                forwards+=1;return lg,hidden[0]
            finally:handle.remove()
        bases=[];zs=[];hs=[]
        for c in prepared:
            lg,h=forward(c['token_ids']);bases.append(lg);hs.append(h)
            with torch.no_grad():enc=sae.encode(torch.tensor(h[None],device=cfg['device']));z=np.zeros(3072);np.add.at(z,enc.top_indices[0].cpu().numpy(),enc.top_acts[0].cpu().numpy())
            zs.append(z);arrays[f'hook_{len(hs)-1}']=h;arrays[f'codes_{len(hs)-1}']=z.astype(np.float32)
        maxnoop=0.0
        for i,c in enumerate(prepared):
            donor=i^1;terms=(zs[i][ids]-zs[donor][ids])*w;scalar=float(terms.sum());natural=scalar*b;hn=float(np.linalg.norm(hs[i].astype(np.float64)));fraction=float(np.linalg.norm(natural))/hn
            scale=min(1.0,cfg['maximum_hook_fraction']/fraction) if fraction else 1.0;delta=natural*scale
            edits={'noop':np.zeros_like(delta),'source_swap':-delta,'orthogonal_random_same_norm':-np.sign(scalar)*np.linalg.norm(delta)*random}
            logits={'baseline':bases[i]}
            for op,d in edits.items():logits[op]=forward(c['token_ids'],d)[0]
            maxnoop=max(maxnoop,float(np.max(abs(logits['noop']-bases[i]))));ps={}
            for op,lg in logits.items():v=np.exp(lg-lg.max());ps[op]=v/v.sum();arrays[f'prob_{i}_{op}']=ps[op].astype(np.float32)
            base=ps['baseline'];itm=items[c['pair_index']];basecontrast=contrast(base,clause,itm);effects={}
            for op,p in ps.items():
                order=np.lexsort((np.arange(len(p)),-abs(p-base)))[:8]
                cm,im=masses(p,clause,itm)
                effects[op]=dict(contrast_delta=contrast(p,clause,itm)-basecontrast,clause_mass=cm,item_mass=im,
                    kl_to_baseline=float(np.sum(p*(np.log(np.maximum(p,1e-300))-np.log(np.maximum(base,1e-300))))),
                    top_changed=[dict(token_id=int(j),token=tokenizer.decode([int(j)]),delta=float(p[j]-base[j])) for j in order])
            components=(zs[i][ids]-mean)*w
            row=dict(c,run_id=cfg['run_id'],metric_version='v1',coordinate=float(components.sum()),positive_weight_component=float(components[w>0].sum()),negative_weight_component=float(components[w<0].sum()),
                donor_case_index=donor,source_difference_coordinate=scalar,natural_hook_fraction=fraction,dose_scale=scale,effects=effects,
                source_terms=[dict(atom=int(j),code_difference=float(zs[i][j]-zs[donor][j]),weight=float(wj),term=float(t)) for j,wj,t in zip(ids,w,terms)])
            rows.append(row)
            with (run/'metrics.raw.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(row,ensure_ascii=False)+'\n')
            if time.perf_counter()-numeric>cfg['numeric_budget_seconds']:raise TimeoutError('Numeric budget exceeded')
        np.savez_compressed(run/'source_arrays.npz',**arrays)
        summary=dict(forwards=forwards,numeric_seconds=time.perf_counter()-numeric,wall_seconds=time.perf_counter()-start,peak_vram_bytes=torch.cuda.max_memory_allocated(),max_noop_absolute_error=maxnoop,source_weight_replay_max_error=weight_error,
            rows=[dict(family=x['family'],topic=x['topic'],conjunction=x['conjunction'],coordinate=x['coordinate'],difference=x['source_difference_coordinate'],dose=x['dose_scale'],source=x['effects']['source_swap']['contrast_delta'],random=x['effects']['orthogonal_random_same_norm']['contrast_delta']) for x in rows],
            metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/run_f4_and_probe.py',generator_script_sha256=code[0]['sha256'])
        write(run/'metrics.summary.json',summary)
        if maxnoop>1e-6:raise ValueError('Noop mismatch')
        write(run/'status.json',dict(status='PASS',forwards=forwards));v=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=v.ok,errors=v.errors))
        if not v.ok:raise ValueError(v.errors)
        print(json.dumps(summary))
    except Exception:
        (run/'stderr.log').write_text(traceback.format_exc());write(run/'status.json',dict(status='FAIL',forwards=forwards));raise


if __name__=='__main__':main()
