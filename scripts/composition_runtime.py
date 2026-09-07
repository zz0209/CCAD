"""Shared actual LM intervention runtime for the authored two-factor panel."""
import json, os, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np
from run_r011s1_raw_hook_asset import ROOT, entry, aggregate, write_json as write
from run_functional_material import logprob
from ccad.artifacts import sha256, validate_run_directory
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor


def realized_delta_norm(delta, positions):
    """A token edited through two named slots receives their vector sum."""
    merged={}
    for position,vector in zip(positions,delta):
        if int(position) not in merged:merged[int(position)]=vector.copy()
        else:merged[int(position)]+=vector
    return float(np.sqrt(sum(np.sum(value**2) for value in merged.values())))


class CompositionRun:
    def __init__(self, config_path, script, sources):
        self.cfg = json.loads(Path(config_path).read_text()); cfg = self.cfg
        self.run = ROOT/'runs'/cfg['run_id']; self.run.mkdir(exist_ok=False)
        self.start = time.perf_counter(); self.inputs = []; self.metrics = []; self.checks = {}; self.forwards = 0
        self.cached_tail_forwards = 0; self.cached_tail_ready = False
        self.script = script; code = []
        write(self.run/'config.resolved.json', cfg)
        for rel in dict.fromkeys([script, 'scripts/composition_runtime.py', 'scripts/run_functional_material.py', 'scripts/run_r011s1_raw_hook_asset.py', 'src/ccad/artifacts.py', 'src/ccad/activation_contract.py']+sources):
            p=ROOT/rel; q=self.run/'source_snapshot'/rel; q.parent.mkdir(parents=True,exist_ok=True); q.write_bytes(p.read_bytes())
            code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path='source_snapshot/'+rel))
        write(self.run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
        write(self.run/'manifest.json',dict(schema_version='composition.function.v1',run_id=cfg['run_id'],run_parent='SEVEN_R3',purpose=cfg['purpose'],milestone='compositional-explanation',evidence_level='controlled_development',started_utc=datetime.now(timezone.utc).isoformat(),project_root=str(ROOT),config_hash=sha256(self.run/'config.resolved.json'),code_snapshot_hash=aggregate(code),source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,mean_constants_source_split='fixed centering cancels in donor differences',threshold_source_split='discovery only; held development contexts are evaluation',statistics_unit='lexical block and time cue, five shared seeds',device='cuda:0',seeds=[s['seed'] for s in cfg['sae_checkpoints']],resource_lease='gpu-0 resource_manager.run',resource_lease_reason=cfg['budget']))
        if cfg.get('evidence_level'):
            manifest=json.loads((self.run/'manifest.json').read_text());manifest.update(evidence_level=cfg['evidence_level'],audit_opened=cfg.get('audit_opened',False),run_parent=cfg.get('run_parent','SEVEN_R3'),threshold_source_split=cfg.get('threshold_source_split','R3 frozen discovery choices; no new fits or outcome selection'));write(self.run/'manifest.json',manifest)
        for fn in ['stdout.log','stderr.log','metrics.raw.jsonl']:(self.run/fn).touch()
        write(self.run/'status.json',dict(status='RUNNING')); self.checked(config_path)

    def checked(self,path):
        p=Path(path); p=p if p.is_absolute() else ROOT/p
        self.inputs.append(entry(p,'existing pinned CCAD input','input')); return p

    def load(self):
        import torch, transformers
        self.torch=torch; self.transformers=transformers; cfg=self.cfg
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision('high'); torch.cuda.reset_peak_memory_stats()
        parent=ROOT/cfg['material_run']; self.checked(parent/'panel.json'); self.checked(parent/'raw_cache.npz')
        self.panel=json.loads((parent/'panel.json').read_text()); self.rows=self.panel['rows']; self.n=len(self.rows)
        self.labels=np.array(self.panel['label_ids']); self.donors={f:np.array([p[f] for p in self.panel['pairs']]) for f in ['number','time','joint']}
        cache=np.load(parent/'raw_cache.npz'); self.h=cache[f"layer{cfg['sae_layer']}"]; self.positions=cache['positions']
        self.discovery=np.array([r['block'] in cfg['discovery_blocks'] and r['cue_id'] in cfg['discovery_cues'] for r in self.rows])
        self.evaluation_ids=np.array([i for i,r in enumerate(self.rows) if 'evaluation_blocks' not in cfg or r['block'] in cfg['evaluation_blocks']])
        modeldir=Path(cfg['model_local_dir'])
        for fn in ['config.json','tokenizer.json','model.safetensors']:self.checked(modeldir/fn)
        self.tokenizer=transformers.AutoTokenizer.from_pretrained(modeldir,local_files_only=True); self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model=transformers.AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().cuda(); self.model.config.use_cache=False
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.checks['token_ids']=all(self.tokenizer.encode(r['text'],add_special_tokens=False)==tr['ids'] for r,tr in zip(self.rows,self.panel['token_rows']))
        self.base=self.evaluate(np.zeros_like(self.h))
        if cfg.get('final_layer_cached_tail'):
            self.validate_cached_tail()
        # Exact repeatability is checked within the selected backend. A cached
        # tail also checks every baseline against the complete model separately.
        noop=self.evaluate(np.zeros_like(self.h))
        if self.cached_tail_ready:
            self.checks['cached_tail_all_baseline_agreement']=bool(np.max(np.abs(noop-self.base))<=1e-4)
            self.checks['noop_exact']=np.array_equal(self.evaluate(np.zeros_like(self.h)),noop)
            assert self.checks['cached_tail_all_baseline_agreement']
        else:self.checks['noop_exact']=np.array_equal(noop,self.base)
        self.progress('READY',n=self.n)

    def forward(self,ids,delta,gradient_slot=None,objective_indices=None):
        torch=self.torch
        if time.perf_counter()-self.start>self.cfg['budget_seconds']:raise TimeoutError('Composition run measured wall budget exhausted')
        if self.cached_tail_ready and gradient_slot is None:
            return self.cached_tail_forward(ids,delta)
        enc=self.tokenizer([self.rows[i]['text'] for i in ids],add_special_tokens=False,padding=True,return_tensors='pt').to('cuda'); ix=torch.arange(len(ids),device='cuda'); last=enc.attention_mask.sum(1)-1
        contract=HookPointContract(f"gpt_neox.layers.{self.cfg['sae_layer']}",self.cfg['sae_layer'],'resid_post',self.model.config.hidden_size)
        leaf=torch.zeros((len(ids),self.h.shape[-1]),device='cuda',requires_grad=gradient_slot is not None)
        def hook(mod,args,out):
            h=extract_primary_hook_tensor(out,contract).clone()
            for slot in range(self.h.shape[1]):
                update=torch.as_tensor(delta[:,slot],device='cuda',dtype=h.dtype)
                if slot==gradient_slot:update=update+leaf
                h[ix,torch.as_tensor(self.positions[ids,slot],device='cuda')]+=update
            return replace_primary_hook_tensor(out,h,contract)
        handle=self.model.get_submodule(contract.module_path).register_forward_hook(hook) if hasattr(contract,'module_path') else self.model.get_submodule(f"gpt_neox.layers.{self.cfg['sae_layer']}").register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(gradient_slot is not None):
                hidden=self.model.gpt_neox(**enc,use_cache=False).last_hidden_state[ix,last]
                if gradient_slot is not None:
                    wanted,original=objective_indices; weight=self.model.get_output_embeddings().weight
                    direction=weight[torch.as_tensor(wanted,device='cuda')]-weight[torch.as_tensor(original,device='cuda')]
                    value=(hidden*direction).sum(); result=torch.autograd.grad(value,leaf)[0].detach().cpu().numpy()
                else:result=logprob(self.model.get_output_embeddings()(hidden).float().cpu().numpy())
        finally:handle.remove()
        self.forwards+=len(ids); return result

    def cached_tail_forward(self,ids,delta):
        """Exact final-block computation; only float32 cache/batch rounding differs."""
        torch=self.torch
        final_slot=self.cached_final_slot
        last=self.positions[ids,final_slot]
        with torch.no_grad():
            h=torch.as_tensor(self.h[ids,final_slot],device='cuda').clone()
            # Preserve the actual hook's sequential float32 additions, including
            # two named changes at the same token. Other positions cannot affect
            # this output after the final attention/MLP block.
            for slot in range(self.h.shape[1]):
                active=self.positions[ids,slot]==last
                if np.any(active):
                    h[torch.as_tensor(active,device='cuda')]+=torch.as_tensor(delta[active,slot],device='cuda',dtype=h.dtype)
            hidden=self.model.gpt_neox.final_layer_norm(h)
            result=logprob(self.model.get_output_embeddings()(hidden).float().cpu().numpy())
        self.forwards+=len(ids); self.cached_tail_forwards+=len(ids)
        return result

    def validate_cached_tail(self):
        assert self.cfg['sae_layer']==self.model.config.num_hidden_layers-1
        last=np.array([len(r['ids'])-1 for r in self.panel['token_rows']])
        slots=[slot for slot in range(self.h.shape[1]) if np.array_equal(self.positions[:,slot],last)]
        assert slots,'Final-token residual is not present in the saved cache.'
        self.cached_final_slot=slots[0]
        ids=np.arange(min(self.cfg['batch_size'],self.n))
        probes=[np.zeros_like(self.h[ids])]
        joint=np.zeros_like(self.h[ids])
        joint[:,0]=.5*(self.h[self.donors['time'][ids],0]-self.h[ids,0])
        joint[:,1]=.5*(self.h[self.donors['number'][ids],1]-self.h[ids,1])
        probes.extend([joint,-joint])
        errors=[]
        for delta in probes:
            full=self.forward(ids,delta)
            tail=self.cached_tail_forward(ids,delta)
            errors.append(float(np.max(np.abs(full-tail))))
        self.checks['cached_tail_full_model_agreement']=max(errors)<=1e-4
        write(self.run/'cached_tail_validation.json',dict(rows=ids.tolist(),full_vocabulary_logprob_max_errors=errors,tolerance=1e-4,
              probes=['zero','half_number_plus_half_time','negative_half_number_plus_half_time'],
              computation='Cached final block residual; original float32 slot additions, final LayerNorm and full unembedding. Gradients and DAS optimization still use the full model.',
              full_model_baseline_all_rows=True))
        assert self.checks['cached_tail_full_model_agreement'],errors
        self.cached_tail_ready=True

    def evaluate(self,delta,ids=None):
        ids=np.arange(self.n) if ids is None else ids;bs=self.cfg['batch_size']
        return np.concatenate([self.forward(ids[off:off+bs],delta[ids[off:off+bs]]) for off in range(0,len(ids),bs)])

    def record(self,row):
        with (self.run/'metrics.raw.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        self.metrics.append(row)

    def measure(self,name,factor,delta,reference=None,**identity):
        ids=self.evaluation_ids;lp=self.evaluate(delta,ids); reference=self.base[self.donors[factor][ids]] if reference is None else reference
        for j,values,ref in zip(ids,lp,reference):
            j=int(j);r=self.rows[j]
            small=values[self.labels]; num=float(np.logaddexp(small[1],small[3])-np.logaddexp(small[0],small[2])); tense=float(np.logaddexp(small[2],small[3])-np.logaddexp(small[0],small[1])); expected=self.rows[self.donors[factor][j]]['expected_label_index']
            self.record(dict(kind='intervention',method=name,factor=factor,row_id=j,block=r['block'],cue_id=r['cue_id'],template=r['template'],number=r['number'],past=r['past'],distractor=r['distractor'],expected=expected,label=int(np.argmax(small)),correct=int(np.argmax(small))==expected,number_logodds=num,past_logodds=tense,label_logprobs=small.tolist(),kl_reference=max(0.,float(np.sum(np.exp(ref)*(ref-values)))),delta_norm=realized_delta_norm(delta[j],self.positions[j]),slot_delta_norm=float(np.linalg.norm(delta[j])),**identity))
        return lp

    def progress(self,stage,**kw):
        result=dict(stage=stage,elapsed=time.perf_counter()-self.start,sequence_forwards=self.forwards,**kw); write(self.run/'progress.json',result); print(json.dumps(result),flush=True)

    def finish(self,error=None):
        self.checks['finite']=all(np.isfinite(v) for r in self.metrics for v in r.values() if isinstance(v,float))
        status='PASS' if error is None and all(self.checks.values()) else 'FAIL'
        env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=self.torch.__version__,transformers=self.transformers.__version__,gpu=self.torch.cuda.get_device_name(),peak_vram_bytes=self.torch.cuda.max_memory_allocated(),cpu_threads=2) if hasattr(self,'model') else {}
        summary=dict(status=status,error=error,checks=self.checks,rows=len(self.metrics),wall_seconds=time.perf_counter()-self.start,sequence_forwards=self.forwards,full_model_sequence_forwards=self.forwards-self.cached_tail_forwards,cached_final_tail_sequence_forwards=self.cached_tail_forwards,scope=self.cfg['scope'],metrics_raw_sha256=sha256(self.run/'metrics.raw.jsonl'),generator_script_path=self.script,generator_script_sha256=sha256(self.run/'source_snapshot'/self.script))
        write(self.run/'inputs.json',dict(inputs=self.inputs)); write(self.run/'environment.json',env); write(self.run/'metrics.summary.json',summary); write(self.run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat())); write(self.run/'stdout.log',summary)
        result=validate_run_directory(self.run); write(self.run/'contract_validation.json',dict(ok=result.ok,errors=list(result.errors))); print(json.dumps(dict(summary=summary,contract_ok=result.ok,errors=list(result.errors))),flush=True)
        return 0 if status=='PASS' and result.ok else 1
