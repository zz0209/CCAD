"""Common-input SAE quality and source-only interpretation candidates.

Reuses the existing quality evaluator; no training, correspondence fitting or audit.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

from run_r006b_topk_capacity import ROOT, evaluate, file_entry, write_json, sha256, HookPointContract
from run_r011s1_raw_hook_asset import aggregate
from ccad.artifacts import validate_run_directory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    cfg_path = ap.parse_args().config
    cfg = json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    run = ROOT / 'runs' / cfg['run_id']
    run.mkdir(exist_ok=False)
    start = time.perf_counter()
    now = datetime.now(timezone.utc).isoformat()
    write_json(run/'config.resolved.json', cfg)
    files = ['scripts/run_material_common_quality.py','scripts/run_r006b_topk_capacity.py',
             'scripts/run_r011s1_raw_hook_asset.py','src/ccad/activation_contract.py',
             'src/ccad/sae_quality.py','src/ccad/artifacts.py']
    code = []
    for rel in files:
        p = ROOT/rel
        dest = run/'source_snapshot'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
        code.append(dict(path=rel,sha256=sha256(p),bytes=p.stat().st_size,snapshot_path=f'source_snapshot/{rel}'))
    write_json(run/'code_hashes.json',dict(files=code,aggregate_sha256=aggregate(code),snapshot_root='source_snapshot'))
    write_json(run/'manifest.json',dict(schema_version='ccad.material.quality.v1',run_id=cfg['run_id'],run_parent='F4',
        purpose=cfg['purpose'],milestone='M4',evidence_level='common_input_asset_comparison_development',started_utc=now,
        project_root=str(ROOT),config_hash=sha256(run/'config.resolved.json'),code_snapshot_hash=aggregate(code),
        source_snapshot_required=True,audit_opened=False,candidate_family_frozen=True,
        mean_constants_source_split='evaluation sample mean for FVE only; no FCC means changed',
        threshold_source_split='deterministic sequence panel; source1 candidates only',
        statistics_unit='shared documents and matched seed identifiers; corpus packages differ',device='cuda:0',seeds=[1,2,3,4,5],
        resource_lease='cpu-heavy -> gpu-0 resource_manager.run',resource_lease_reason=cfg['budget']))
    for name in ('stdout.log','stderr.log','metrics.raw.jsonl'):
        (run/name).touch()
    write_json(run/'status.json',dict(status='RUNNING',updated_utc=now))
    inputs, checks, results, env = [], {}, [], {}
    error = None
    def checked(path, expected=None):
        p = Path(path)
        if not p.is_absolute(): p = ROOT/p
        ident = sha256(p)
        if expected is not None and ident != expected: raise ValueError(f'Identity changed: {p}')
        inputs.append(file_entry(p,'CCAD existing frozen assets','local research','common_quality_input'))
        return p
    def load(path): return json.loads(checked(path).read_text(encoding='utf-8-sig'))
    try:
        checked(cfg_path)
        load('.aris/compute/local-r006b1-env-spec.json')
        assets = [load(p) for p in cfg['asset_configs']]
        base = assets[0]
        corpus = ROOT/'runs'/base['paired_corpus_run']
        docs = [json.loads(s) for s in checked(corpus/'artifacts/documents.jsonl').read_text().splitlines() if s]
        forbidden = []
        for path in cfg['excluded_document_files']:
            p = checked(path)
            data = [json.loads(s) for s in p.read_text().splitlines() if s] if p.suffix == '.jsonl' else json.loads(p.read_text())['documents']
            forbidden.extend(data)
        for field in ('document_id','text_sha256'):
            left = {d[field] for d in docs if d.get(field)}
            right = {d[field] for d in forbidden if d.get(field)}
            checks['no_train_validation_overlap_'+field] = not bool(left & right)
        assert all(checks.values()), 'Evaluation overlaps SAE data'
        tm = load(base['token_manifest_path'])
        ti = tm['outputs']['calibration']
        tokenpath = checked(corpus/ti['path'],ti['sha256'])
        rm = load(Path(base['bulk_output_dir'])/'raw_hook_manifest.json')
        rawmeta = next(r for r in rm['splits'] if r['split']=='calibration')
        rawpath = checked(rawmeta['path'],rawmeta['sha256'])
        seqmeta = load(corpus/'artifacts/sequence_records.json')['sequences']
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SPARSIFY_DISABLE_TRITON='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        sys.path[:0] = [base['sparsify_source_dir'],base['sparsify_overlay_dir']]
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sparsify.sparse_coder import SparseCoder
        torch.set_num_threads(cfg['cpu_threads'])
        torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()
        ids = cfg['sequence_ids']
        assert len(ids)==len(set(ids))==64 and max(ids)<ti['sequences']
        tokens_all = np.memmap(tokenpath,dtype='<u2',mode='r').reshape(-1,128)
        tokens = np.array(tokens_all[ids],dtype=np.int64)
        raw = np.memmap(rawpath,dtype='<f4',mode='r',shape=tuple(rawmeta['shape']))
        rowids = (np.array(ids)[:,None]*128+np.arange(128)[None,:]).ravel()
        hook = np.array(raw[rowids])
        dataset = [dict(input_ids=torch.tensor(row)) for row in tokens]
        selected_docs = sorted({d for s in seqmeta if s['sequence_index'] in ids for d in s['document_ids']})
        write_json(run/'selected_inputs.json',dict(sequence_ids=ids,document_ids=selected_docs,token_rows=8192,
                   selection='every eighth sequence of existing512 panel; fixed before evaluation'))
        model = AutoModelForCausalLM.from_pretrained(base['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to('cuda:0')
        model.config.use_cache=False
        tokenizer = AutoTokenizer.from_pretrained(base['model_local_dir'],local_files_only=True)
        contract = HookPointContract(base['hook_module_path'],5,'resid_post',768)
        captured = {}
        handle = model.get_submodule(base['hook_module_path']).register_forward_hook(lambda m,i,o: captured.update(h=o[0] if isinstance(o,tuple) else o))
        with torch.no_grad(): model(torch.tensor(tokens[:4],device='cuda:0'),use_cache=False)
        handle.remove()
        replay = float(np.linalg.norm(captured['h'].detach().cpu().numpy().reshape(-1,768)-hook[:512])/np.linalg.norm(hook[:512]))
        checks['cached_hook_replay'] = replay < 1e-4
        del captured
        for ai, asset in enumerate(assets):
            label = 'short' if ai==0 else 'long'
            for item in asset['saes']:
                path = checked(Path(item['path'])/'sae.safetensors',item['sha256'])
                checked(path.parent/'cfg.json')
                sae = SparseCoder.load_from_disk(path.parent,device='cuda:0').eval()
                tick = time.perf_counter()
                q = evaluate(model,sae,model.get_submodule(base['hook_module_path']),contract,6,dataset,cfg['batch_size'],'cuda:0',torch)
                counts = np.zeros(3072,dtype=np.int64)
                top = [[] for _ in range(3072)] if label=='long' and item['seed']==1 else None
                with torch.no_grad():
                    for off in range(0,len(hook),512):
                        enc=sae.encode(torch.tensor(hook[off:off+512],device='cuda:0'))
                        ii,aa=enc.top_indices.cpu().numpy(),enc.top_acts.cpu().numpy()
                        nz=aa!=0
                        counts += np.bincount(ii[nz],minlength=3072)
                        if top is not None:
                            for j,(inds,acts) in enumerate(zip(ii,aa)):
                                for atom,act in zip(inds,acts):
                                    if act>0: top[int(atom)].append((float(act),off+j))
                row=dict(configuration=label,seed=item['seed'],quality=q,seconds=time.perf_counter()-tick,
                         firing_counts=counts.tolist(),cached_firing_total_matches=int(counts.sum())==q['feature_firing_count_distribution']['total_firings'])
                results.append(row)
                with (run/'metrics.raw.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                print(json.dumps({k:row[k] for k in ('configuration','seed','seconds')}|dict(fve=q['fve'],ce_recovered=q['ce_recovered'],alive=q['alive_features'])),flush=True)
                if top is not None:
                    candidates=[]
                    for atom,hits in enumerate(top):
                        hits=sorted(hits,reverse=True)[:24]
                        if len(hits)<12: continue
                        common=Counter(int(tokens[r//128,r%128]) for act,r in hits).most_common(1)[0]
                        tok=tokenizer.decode([common[0]])
                        if not any(c.isalpha() for c in tok): continue
                        sequences=len({r//128 for act,r in hits})
                        if sequences<4: continue
                        examples=[]
                        for act,r in hits[:8]:
                            s,p=divmod(r,128)
                            examples.append(dict(sequence=ids[s],position=p,activation=act,token=tokenizer.decode([int(tokens[s,p])]),
                                left=tokenizer.decode(tokens[s,max(0,p-16):p].tolist()),right=tokenizer.decode(tokens[s,p+1:min(128,p+17)].tolist())))
                        candidates.append(dict(atom=atom,frequency=int(counts[atom]),dominant_token=tok,top24_token_fraction=common[1]/len(hits),
                            top24_sequences=sequences,score=common[1]/len(hits)*math_log(sequences),examples=examples))
                    candidates.sort(key=lambda r:(-r['score'],r['atom']))
                    write_json(run/'SOURCE1_CANDIDATES.json',dict(scope='source1 only lexical concentration opportunities, not semantic validation; all3072 frequencies retained',candidates=candidates[:32]))
                del sae
                if time.perf_counter()-start>cfg['wall_budget_seconds']: raise TimeoutError('Common quality budget exceeded')
        checks.update(all_ten=len(results)==10,quality_hook_checks=all(r['quality']['hook_oracle_max_error']==0 and r['quality']['capture_logit_max_error']==0 for r in results),
                      clean_baseline_same=max(r['quality']['ce']['clean'] for r in results)-min(r['quality']['ce']['clean'] for r in results)==0,
                      zero_baseline_same=max(r['quality']['ce']['zero'] for r in results)-min(r['quality']['ce']['zero'] for r in results)==0)
        env=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),peak_vram_allocated=torch.cuda.max_memory_allocated())
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        (run/'stderr.log').write_text(traceback.format_exc())
    status='PASS' if not error and checks and all(checks.values()) else 'FAIL'
    write_json(run/'inputs.json',dict(inputs=inputs))
    write_json(run/'environment.json',env)
    summary=dict(status=status,error=error,checks=checks,wall_seconds=time.perf_counter()-start,saes=len(results),model_forwards=1+len(results)*64,
                 metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),scope=cfg['scope'])
    write_json(run/'metrics.summary.json',summary)
    write_json(run/'stdout.log',summary)
    write_json(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()))
    valid=validate_run_directory(run)
    write_json(run/'contract_validation.json',dict(ok=valid.ok,errors=list(valid.errors)))
    print(json.dumps(dict(summary=summary,contract_ok=valid.ok,contract_errors=list(valid.errors))),flush=True)
    return 0 if status=='PASS' and valid.ok else 1


def math_log(value):
    import math
    return math.log1p(value)


if __name__=='__main__':
    raise SystemExit(main())
