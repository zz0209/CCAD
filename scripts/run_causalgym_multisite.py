"""Compare source interventions across aligned positions in CausalGym train.

The same full-model consumer is also reusable by the subsequent group experiment.
Raw intervention coverage and SAE material fidelity are separate observations.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import runpy
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
import numpy as np
from run_causalgym_native_transfer import prepare_panel
from run_r011s1_raw_hook_asset import ROOT, entry, aggregate, write_json as write
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor
from ccad.artifacts import sha256, validate_run_directory


def aligned_positions(row, donor, mode):
    """Explicit recipient/donor token pairs; absent full-length control is retained."""
    n, nd = len(row['tokens']), len(donor['tokens'])
    ends, dend = row['region_ends'], donor['region_ends']
    if mode.startswith('region_') and mode != 'region_ends_nonfinal':
        region = int(mode.split('_')[1])
        pairs = [(ends[region], dend[region])]
    elif mode == 'region_ends_nonfinal':
        pairs = [(p, q) for p, q in zip(ends, dend) if 0 <= p < n-1 and 0 <= q < nd-1]
    elif mode in ['suffix_nonfinal', 'suffix_all']:
        c = row['changed_region']
        start, dstart = ends[c], dend[c]
        # After the last altered aligned region, the token suffix must agree.
        if row['tokens'][start+1:] != donor['tokens'][dstart+1:]:
            raise ValueError('Unchanged aligned suffix has different tokenization')
        stop = n if mode == 'suffix_all' else n-1
        pairs = [(p, dstart+p-start) for p in range(start, stop)]
    elif mode == 'full_tokens_equal_length':
        if n != nd:
            return None
        pairs = [(p, p) for p in range(n)]
    else:
        raise ValueError(mode)
    unique = {}
    for p, q in pairs:
        if p < 0 or q < 0:
            continue
        if p >= n or q >= nd:
            raise ValueError('Alignment leaves a real token sequence')
        if p in unique and unique[p] != q:
            raise ValueError('Duplicate recipient position has inconsistent donors')
        unique[p] = q
    return sorted(unique.items())


class MultisiteWork:
    def __init__(self, cfg, config_path, source_files=None):
        self.cfg, self.config_path = cfg, Path(config_path)
        self.run = ROOT/'runs'/cfg['run_id']
        self.run.mkdir(exist_ok=False)
        self.started = datetime.now(timezone.utc)
        self.wall_start, self.cpu_start = time.perf_counter(), time.process_time()
        self.inputs, self.metrics, self.checks, self.environment = [], [], {}, {}
        self.sequence_forwards = self.token_forwards = 0
        write(self.run/'config.resolved.json', cfg)
        sources = source_files or ['scripts/run_causalgym_multisite.py', 'scripts/run_causalgym_native_transfer.py',
                   'scripts/run_r011s1_raw_hook_asset.py', 'src/ccad/activation_contract.py', 'src/ccad/artifacts.py',
                   'src/ccad/factor_correspondence.py', 'src/ccad/native_operation.py', 'src/ccad/nip_baselines.py']
        code = []
        for rel in sources:
            p = ROOT/rel
            dest = self.run/'source_snapshot'/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
            code.append(dict(path=rel, sha256=sha256(p), bytes=p.stat().st_size, snapshot_path='source_snapshot/'+rel))
        self.driver = cfg.get('generator_script', 'scripts/run_causalgym_multisite.py')
        write(self.run/'code_hashes.json', dict(files=code, aggregate_sha256=aggregate(code), snapshot_root='source_snapshot'))
        write(self.run/'manifest.json', dict(schema_version='causalgym.multisite.v1', run_id=cfg['run_id'],
              run_parent=cfg.get('run_parent','FINAL_FIVE_R15'), purpose=cfg['purpose'], milestone=cfg.get('milestone','external-multisite-source-and-native-groups'),
              evidence_level='controlled_development', started_utc=self.started.isoformat(), project_root=str(ROOT),
              config_hash=sha256(self.run/'config.resolved.json'), code_snapshot_hash=aggregate(code), source_snapshot_required=True,
              git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              audit_opened=cfg['audit_opened'], candidate_family_frozen=cfg['candidate_family_frozen'],
              mean_constants_source_split='Same dictionary donor differences cancel a fixed mean and decoder bias',
              threshold_source_split='Configuration before this experiment; original exposed train components remain development',
              statistics_unit=cfg.get('statistics_unit','prompt-connected components, reciprocal directions and shared SAE seeds'),
              device=cfg['device'], seeds=cfg['seeds'], resource_lease='gpu-0' if cfg['device'].startswith('cuda') else 'cpu-heavy',
              resource_lease_reason=cfg['budget'], model_revision=cfg['model_revision'], dataset_revision=cfg['dataset_revision']))
        for name in ['stdout.log','stderr.log','metrics.raw.jsonl']:
            (self.run/name).touch()
        write(self.run/'status.json', dict(status='RUNNING', started_at_utc=self.started.isoformat()))
        self.checked(self.config_path, 'CCAD experiment configuration')

    def checked(self, path, source='Existing pinned CCAD asset', boundary='internal'):
        p = Path(path)
        self.inputs.append(entry(p, source, 'actual input', boundary))
        write(self.run/'inputs.json', dict(inputs=self.inputs))
        return p

    def progress(self, stage, **values):
        item = dict(stage=stage, written_at_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed=time.perf_counter()-self.wall_start, process_cpu_seconds=time.process_time()-self.cpu_start,
                    sequence_forwards=self.sequence_forwards, token_forwards=self.token_forwards, **values)
        write(self.run/'progress.json', item)
        print(json.dumps(item), flush=True)

    def record(self, **row):
        row = dict(run_id=self.cfg['run_id'], metric_version='multisite-v1', **row)
        with (self.run/'metrics.raw.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        self.metrics.append(row)

    def setup(self):
        import torch
        import transformers
        self.torch = torch
        cfg = self.cfg
        torch.set_num_threads(cfg['cpu_threads'])
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        self.device = torch.device(cfg['device'])
        if self.device.type=='cuda':
            torch.cuda.set_device(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.environment = dict(python=sys.executable, python_version=platform.python_version(), os=platform.platform(),
            numpy=np.__version__, torch=torch.__version__, transformers=transformers.__version__,
            device=str(self.device), cpu_threads=torch.get_num_threads(), matmul_precision=torch.get_float32_matmul_precision(),
            gpu=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else 'not_used')
        write(self.run/'environment.json', self.environment)
        kernel = self.checked(Path(cfg['sparsify_source'])/'sparsify/fused_encoder.py', 'Sparsify42c0645 encoding kernel', 'MIT')
        self.checked(Path(cfg['sparsify_source'])/'sparsify/sparse_coder.py', 'Sparsify42c0645 preprocessing reference', 'MIT')
        self.encode_kernel = runpy.run_path(str(kernel))['fused_encoder']
        self.checked(ROOT/'.aris/compute/local-r006b1-env-spec.json')
        if cfg.get('prepared_panel'):
            dataset=self.checked(Path(cfg['prepared_panel']),cfg['dataset_name']+' prepared panel','Pinned public task data')
            prepared=json.loads(dataset.read_text())
            panel,inventory=prepared['rows'],prepared['selection_inventory']
            for reference in prepared.get('source_files',[]):
                self.checked(Path(reference),cfg['dataset_name']+' original input','Pinned public task data')
        else:
            dataset = self.checked(Path(cfg['dataset_dir'])/'train.json','aryaman/causalgym '+cfg['dataset_revision'],'MIT task data')
            self.checked(Path(cfg['dataset_dir'])/'README.md','Pinned dataset card','MIT')
            panel, inventory = prepare_panel(json.loads(dataset.read_text()), cfg)
        write(self.run/'selection_inventory.json', inventory)
        modeldir = Path(cfg['model_local_dir'])
        for name in ['config.json','tokenizer.json','model.safetensors']:
            self.checked(modeldir/name,'EleutherAI/pythia-1b-deduped '+cfg['model_revision'],'Apache-2.0')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = transformers.AutoModelForCausalLM.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,
                        dtype=torch.float32,attn_implementation='eager').eval().to(self.device)
        self.model.config.use_cache = False
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.dim = self.model.config.hidden_size
        self.contract = HookPointContract(f"gpt_neox.layers.{cfg['layer']}",cfg['layer'],'resid_post',self.dim)
        self.module = self.model.get_submodule(f"gpt_neox.layers.{cfg['layer']}")
        self.label_ids = {}
        for row in panel:
            tokens = self.tokenizer.encode(row['text'],add_special_tokens=False)
            ends, prefix = [], ''
            for span in row['spans']:
                prefix += span
                pt = self.tokenizer.encode(prefix,add_special_tokens=False)
                if pt != tokens[:len(pt)]:
                    raise ValueError('Retokenized region boundary')
                ends.append(len(pt)-1)
            for label in [row['label'],row['donor_label']]:
                ids = self.tokenizer.encode(label,add_special_tokens=False)
                if len(ids)!=1 or self.tokenizer.encode(row['text']+label,add_special_tokens=False)!=tokens+ids:
                    raise ValueError('Non-concatenative continuation token')
                self.label_ids[label] = ids[0]
            row.update(tokens=tokens,region_ends=ends)
        self.panel = panel
        self.max_length = max(len(row['tokens']) for row in panel)
        self.donors = np.asarray([r['donor_id'] for r in panel])
        self.hidden = np.zeros((len(panel),self.max_length,self.dim),np.float32)
        self.baseline = np.zeros((len(panel),self.model.config.vocab_size),np.float64)
        write(self.run/'panel.json',dict(rows=panel,label_ids=self.label_ids,dataset_split='train'))
        for ids in self.batches(np.arange(len(panel))):
            lp, hidden = self.forward(ids,capture=True)
            self.baseline[ids] = lp
            self.hidden[ids] = hidden
        np.savez_compressed(self.run/'raw_cache.npz',hidden=self.hidden,baseline_logprobs=self.baseline.astype(np.float32))
        probe = np.arange(min(cfg['batch_size'],len(panel)))
        lp, hidden = self.forward(probe,np.zeros_like(self.hidden[probe]),capture=True)
        self.checks['same_batch_noop_exact'] = bool(np.array_equal(lp,self.baseline[probe]))
        self.checks['captured_states_exact'] = bool(np.array_equal(hidden,self.hidden[probe]))
        self.checks['prompt_component_split'] = True
        self.progress('MATERIAL_CAPTURED',rows=len(panel),max_length=self.max_length)

    def batches(self, ids):
        for offset in range(0,len(ids),self.cfg['batch_size']):
            yield np.asarray(ids[offset:offset+self.cfg['batch_size']],int)

    def forward(self, row_ids, delta=None, capture=False, gradient=False, differentiable=False, replace_states=False):
        torch = self.torch
        if time.perf_counter()-self.wall_start > self.cfg['budget_seconds']:
            raise TimeoutError('Measured experiment wall budget exhausted')
        enc = self.tokenizer([self.panel[i]['text'] for i in row_ids],add_special_tokens=False,padding=True,return_tensors='pt').to(self.device)
        ix = torch.arange(len(row_ids),device=self.device)
        last = enc.attention_mask.sum(1)-1
        width = enc.input_ids.shape[1]
        captured = []
        leaf = torch.zeros((len(row_ids),width,self.dim),device=self.device,requires_grad=gradient)
        def hook(mod,args,output):
            h = extract_primary_hook_tensor(output,self.contract)
            if capture:
                cap = np.zeros((len(row_ids),self.max_length,self.dim),np.float32)
                real = h.detach().cpu().numpy()
                for j,i in enumerate(row_ids):
                    cap[j,:len(self.panel[i]['tokens'])] = real[j,:len(self.panel[i]['tokens'])]
                captured.append(cap)
            if delta is not None:
                update = torch.as_tensor(delta,device=self.device,dtype=h.dtype)[:,:width]
                if gradient:
                    update = update+leaf
                return replace_primary_hook_tensor(output,update if replace_states else h+update,self.contract)
            return output
        handle = self.module.register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(gradient or differentiable):
                out = self.model.gpt_neox(**enc,use_cache=False).last_hidden_state
                logits = self.model.get_output_embeddings()(out[ix,last])
                # Float64 normalization avoids an O(1e-7) signed normalization
                # error dominating genuinely tiny full-vocabulary KL values.
                live = torch.log_softmax(logits.to(torch.float64),dim=-1)
                lp = None if differentiable else live.detach().cpu().numpy().astype(np.float64)
                derivative = None
                if gradient:
                    a = torch.as_tensor([self.label_ids[self.panel[i]['label']] for i in row_ids],device=self.device)
                    b = torch.as_tensor([self.label_ids[self.panel[i]['donor_label']] for i in row_ids],device=self.device)
                    g = torch.autograd.grad((logits[ix,b]-logits[ix,a]).sum(),leaf)[0].detach().cpu().numpy()
                    derivative = np.zeros((len(row_ids),self.max_length,self.dim),np.float64)
                    derivative[:,:width] = g
        finally:
            handle.remove()
        self.sequence_forwards += len(row_ids)
        self.token_forwards += int(enc.attention_mask.sum())
        return (live if differentiable else lp), derivative if gradient else captured[0] if captured else None

    def load_sae(self, seed, positions=None):
        from safetensors import safe_open
        torch = self.torch
        p = Path(self.cfg['sae_root'])/f'seed_{seed}'
        scfg = json.loads(self.checked(p/'cfg.json').read_text())
        if scfg['transcode'] or scfg['skip_connection']:
            raise ValueError('Requires ordinary residual SAE')
        with safe_open(self.checked(p/'sae.safetensors'),framework='pt',device=str(self.device)) as f:
            ew,eb,db,dw = [f.get_tensor(k) for k in ['encoder.weight','encoder.bias','b_dec','W_dec']]
        if positions is not None:
            positions=np.asarray(positions,dtype=int)
            if positions.shape!=(len(self.panel),) or any(p<0 or p>=len(r['tokens']) for p,r in zip(positions,self.panel)):
                raise ValueError('Requested SAE positions must be real prompt tokens')
        codes = np.zeros((len(self.panel),self.max_length,ew.shape[0]) if positions is None else (len(self.panel),ew.shape[0]),np.float32)
        for ids in self.batches(np.arange(len(self.panel))):
            selected=self.hidden[ids] if positions is None else self.hidden[ids,positions[ids]]
            flat = torch.as_tensor(selected.reshape(-1,self.dim),device=self.device)
            with torch.no_grad():
                act,index,_ = self.encode_kernel(flat-db,ew,eb,scfg['k'],scfg['activation'])
                dense = torch.zeros((len(flat),ew.shape[0]),device=self.device).scatter_(1,index,act)
                codes[ids] = dense.cpu().numpy().reshape(codes[ids].shape)
        if positions is None:
            for i,row in enumerate(self.panel):
                codes[i,len(row['tokens']):] = 0
        np.savez_compressed(self.run/f'seed{seed}_codes.npz',codes=codes,**({} if positions is None else dict(positions=positions)))
        return dict(seed=seed,codes=codes,decoder=dw.detach().cpu().numpy(),cfg=scfg)

    def alignment(self, ids, mode):
        records = [aligned_positions(self.panel[i],self.panel[self.donors[i]],mode) for i in ids]
        valid = np.asarray([r is not None and bool(r) for r in records])
        return valid, records

    def delta(self, ids, mode, states, decoder=None, support=None):
        torch = self.torch
        valid, alignments = self.alignment(ids,mode)
        delta = np.zeros((len(ids),self.max_length,self.dim),np.float32)
        d_tensor = None
        if decoder is not None:
            d_tensor = torch.as_tensor(decoder if support is None else decoder[support],device=self.device)
        for j,i in enumerate(ids):
            if not valid[j]:
                continue
            pp,qq = np.asarray(alignments[j]).T
            dz = states[self.donors[i],qq]-states[i,pp]
            if decoder is not None:
                if support is not None:
                    dz = dz[:,support]
                with torch.no_grad():
                    dz = (torch.as_tensor(dz,device=self.device)@d_tensor).cpu().numpy()
            delta[j,pp] = dz
        return delta,valid,alignments

    def measure(self, ids, mode, method, delta, seed=0, reference=None, **extra):
        outputs = np.zeros((len(ids),self.baseline.shape[1]),np.float64)
        for local in self.batches(np.arange(len(ids))):
            outputs[local],_ = self.forward(ids[local],delta[local])
        for j,i in enumerate(ids):
            row = self.panel[i]
            a,b = self.label_ids[row['label']],self.label_ids[row['donor_label']]
            before = self.baseline[i,b]-self.baseline[i,a]
            margin = outputs[j,b]-outputs[j,a]
            donor = self.baseline[self.donors[i]]
            ref = donor if reference is None else reference[j]
            kl = float(np.sum(np.exp(ref)*(ref-outputs[j])))
            self.record(kind='intervention',task=row['task'],row_id=int(i),component=row['component'],pair_key=row['pair_key'],
                split=row['split'],mode=mode,method=method,seed=seed,donor_oriented_change=float(margin-before),
                donor_oriented_margin=float(margin),donor_label_correct=bool(margin>0),base_label_correct=bool(before<0),
                donor_baseline_correct=bool(donor[b]>donor[a]),kl_to_reference=max(0.,kl),
                reference_identity='natural_donor' if reference is None else extra.get('reference_identity','specified_source_operation'),
                edit_norm=float(np.linalg.norm(delta[j])),positions_edited=int(np.any(delta[j]!=0,axis=1).sum()),
                **{k:v for k,v in extra.items() if k!='reference_identity'})
        return outputs

    def finish(self, error=None):
        if error:
            (self.run/'stderr.log').write_text(error,encoding='utf-8')
        rows = [json.loads(x) for x in (self.run/'metrics.raw.jsonl').read_text().splitlines()]
        self.checks['finite_results'] = all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v,float))
        self.checks['unique_results'] = len(rows)==len({tuple(r.get(k) for k in ['kind','task','row_id','mode','method','seed','target_seed','operation']) for r in rows})
        status = 'PASS' if error is None and rows and all(self.checks.values()) else 'FAIL'
        groups = {}
        for row in rows:
            key = tuple(row.get(k) for k in ['task','mode','method','seed','target_seed','operation','split'])
            groups.setdefault(key,[]).append(row)
        cells = []
        for key,rr in groups.items():
            cell = dict(zip(['task','mode','method','seed','target_seed','operation','split'],key))
            cell.update(n=len(rr),components=len({r['component'] for r in rr}))
            for name in ['donor_oriented_change','donor_label_correct','base_label_correct','donor_baseline_correct','kl_to_reference','edit_norm','positions_edited']:
                vv = [r[name] for r in rr if name in r]
                if vv:
                    cell[name+'_mean'] = float(np.mean(vv))
            cells.append(cell)
        summary = dict(status=status,error=error,checks=self.checks,rows=len(rows),cells=cells,
            wall_seconds=time.perf_counter()-self.wall_start,process_cpu_seconds=time.process_time()-self.cpu_start,
            sequence_forwards=self.sequence_forwards,token_forwards=self.token_forwards,
            metrics_raw_sha256=sha256(self.run/'metrics.raw.jsonl'),generator_script_path=self.driver,
            generator_script_sha256=sha256(self.run/'source_snapshot'/self.driver),scope=self.cfg['scope'])
        if hasattr(self,'device') and self.device.type=='cuda':
            summary['peak_allocated_bytes'] = self.torch.cuda.max_memory_allocated(self.device)
        write(self.run/'metrics.summary.json',summary)
        write(self.run/'inputs.json',dict(inputs=self.inputs))
        write(self.run/'environment.json',self.environment)
        write(self.run/'stdout.log',summary)
        write(self.run/'status.json',dict(status=status,error=error,started_at_utc=self.started.isoformat(),ended_at_utc=datetime.now(timezone.utc).isoformat()))
        contract = validate_run_directory(self.run)
        write(self.run/'contract_validation.json',dict(ok=contract.ok,errors=list(contract.errors)))
        print(json.dumps(dict(status=status,contract_ok=contract.ok,wall_seconds=summary['wall_seconds'],rows=len(rows),error=error)),flush=True)
        return 0 if status=='PASS' and contract.ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    w = MultisiteWork(cfg,args.config)
    error = None
    try:
        w.setup()
        modes_by_task, coverage = {}, []
        for task in cfg['tasks']:
            ids = np.asarray([i for i,r in enumerate(w.panel) if r['task']==task])
            regions = len(w.panel[ids[0]]['region_ends'])
            modes_by_task[task] = [f'region_{k}' for k in range(1,regions)]+cfg['combined_modes']
            for mode in modes_by_task[task]:
                delta,valid,alignments = w.delta(ids,mode,w.hidden)
                coverage.append(dict(task=task,mode=mode,all_rows=ids.tolist(),eligible_rows=ids[valid].tolist(),
                                     ineligible_rows=ids[~valid].tolist(),alignment=alignments))
                if valid.any():
                    lp = w.measure(ids[valid],mode,'raw_donor',delta[valid])
                    if mode=='full_tokens_equal_length':
                        difference = float(np.max(np.abs(lp-w.baseline[w.donors[ids[valid]]])))
                        reference = w.baseline[w.donors[ids[valid]]]
                        kl = np.sum(np.exp(reference)*(reference-lp),axis=1)
                        replacement_error = []
                        # A same-batch exact-state replay is the indexing/hook
                        # invariant; cached subtract/add additionally rounds at
                        # float32 and can differ across GEMM batch shapes.
                        for bid in w.batches(ids[valid]):
                            donor_lp, donor_h = w.forward(w.donors[bid],capture=True)
                            copied_lp,_ = w.forward(bid,donor_h,replace_states=True)
                            replacement_error.append(float(np.max(np.abs(donor_lp-copied_lp))))
                        w.checks['same_batch_exact_state_counterfactual_'+task] = max(replacement_error)<1e-10
                        w.checks['cached_float32_counterfactual_kl_'+task] = bool(np.max(kl)<cfg['full_sequence_kl_tolerance'])
                        write(w.run/f'{task}_full_counterfactual_check.json',dict(max_logprob_error=difference,
                            max_normalized_kl=float(np.max(kl)),same_batch_exact_replacement_error=max(replacement_error),
                            rows=int(valid.sum()),scope='Exact-state indexing replay and float32 cached-add numerical discrepancy are distinct checks; v1 absolute-tail-logprob threshold failure is retained.'))
            w.progress('RAW_COVERAGE_COMPLETE',task=task)
        write(w.run/'position_coverage.json',dict(records=coverage,selection='Token/region identity only, no outcomes used'))
        for seed in cfg['seeds']:
            sae = w.load_sae(seed)
            for task in cfg['tasks']:
                ids = np.asarray([i for i,r in enumerate(w.panel) if r['task']==task])
                for mode in modes_by_task[task]:
                    delta,valid,_ = w.delta(ids,mode,sae['codes'],sae['decoder'])
                    if valid.any():
                        w.measure(ids[valid],mode,'full_sae_donor',delta[valid],seed=seed)
                w.progress('SAE_COVERAGE_COMPLETE',task=task,seed=seed)
            del sae
            gc.collect()
        w.checks['all_configured_seed_materials'] = len(list(w.run.glob('seed*_codes.npz')))==len(cfg['seeds'])
    except Exception:
        error = traceback.format_exc()
    return w.finish(error)


if __name__=='__main__':
    raise SystemExit(main())
