"""Replay frozen source edits and expose probability mass, without target inference."""
from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np
from run_r011s1_raw_hook_asset import ROOT, entry, aggregate, write_json as write
from ccad.artifacts import sha256, validate_run_directory
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor
from f4_probability_endpoints import log_prob, probability_metrics, endpoint_positions


def probability_order(baseline, source, limit):
    """Stable IDs break ties; signed probability change, not extreme logit change."""
    pb, ps = np.exp(log_prob(baseline)), np.exp(log_prob(source))
    ids = np.arange(pb.shape[1])
    return pb, ps, [dict(base=np.lexsort((ids, -b))[:limit],
                         source=np.lexsort((ids, -s))[:limit],
                         increase=np.lexsort((ids, -(s-b)))[:limit],
                         decrease=np.lexsort((ids, s-b))[:limit]) for b, s in zip(pb, ps)]


def case_key(r):
    return tuple(r[k] for k in ('source_seed', 'source_atom', 'condition', 'sequence', 'donor_sequence'))


def source_anchor(p):
    # source_to_candidate_* starts with source_ but is method-dependent.
    return {k:p[k] for k in ('source_nll_deltas','source_to_baseline_kl','positions','observed_next_token_ids')}


def observed_next_id(tokens, position):
    return int(tokens[position+1]) if position+1<len(tokens) and tokens[position]!=0 and tokens[position+1]!=0 else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True); args = ap.parse_args()
    cfg = json.loads(args.config.read_text()); run = ROOT/'runs'/cfg['run_id']; run.mkdir(exist_ok=False)
    started = time.perf_counter(); write(run/'config.resolved.json', cfg)
    for name in ('stdout.log', 'stderr.log', 'metrics.raw.jsonl'): (run/name).touch()
    code = []
    for rel in ('scripts/replay_f4_source_probability.py', 'scripts/run_r011s1_raw_hook_asset.py',
                'scripts/f4_probability_endpoints.py', 'src/ccad/artifacts.py', 'src/ccad/activation_contract.py'):
        p = ROOT/rel; dst = run/'source_snapshot'/rel; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(p.read_bytes())
        code.append(dict(path=rel, sha256=sha256(p), bytes=p.stat().st_size, snapshot_path=f'source_snapshot/{rel}'))
    write(run/'code_hashes.json', dict(files=code, aggregate_sha256=aggregate(code), snapshot_root='source_snapshot'))
    write(run/'manifest.json', dict(schema_version='fcc.source.probability.replay.v1', run_id=cfg['run_id'], run_parent='F4',
        purpose=cfg['purpose'], milestone='M4', evidence_level='exposed_source_development_replay',
        started_utc=datetime.now(timezone.utc).isoformat(), project_root=str(ROOT), config_hash=sha256(run/'config.resolved.json'),
        code_snapshot_hash=aggregate(code), source_snapshot_required=True, audit_opened=False, candidate_family_frozen=True,
        mean_constants_source_split='frozen parent; cancels in donor difference', threshold_source_split='frozen parent source-only',
        statistics_unit='dependent document/query cases; descriptive source inspection', device=cfg['device'], seeds=[1,2,3,4,5],
        resource_lease='gpu-0 via resource_manager.run', resource_lease_reason='22 bounded forwards on existing model, no disk lease'))
    write(run/'status.json', dict(status='RUNNING')); inputs=[]; checks={}; rows=[]; arrays={}; forwards=0; timers={}; env={}; error=None
    def checked(p, expected=None):
        p = Path(p); p = p if p.is_absolute() else ROOT/p
        info = entry(p, 'frozen parent artifact', 'replay_input')
        if expected and info['sha256'] != expected: raise ValueError(f'Input identity mismatch: {p}')
        inputs.append(info); return p
    try:
        lease=json.loads(subprocess.check_output([sys.executable,str(ROOT.parent/'.resource_manager/resource_manager.py'),'status','--resource','gpu-0'],text=True))
        write(run/'resource_status_at_start.json',lease)
        cases=[]
        common_model=None
        for parent in cfg['parents']:
            root=ROOT/parent['run']; old=json.loads(checked(root/'config.resolved.json',parent['config_sha256']).read_text())
            model_identity=tuple(old[k] for k in ('model_local_dir','model_revision','hook_module_path','hook_hidden_size','context_length'))
            if common_model is not None and model_identity!=common_model: raise ValueError('Parent model/hook identity differs')
            common_model=model_identity
            details=checked(root/'case_details.jsonl',parent['details_sha256']); metrics=checked(root/'metrics.raw.jsonl',parent['metrics_sha256'])
            refs={}
            for line in metrics.read_text().splitlines():
                r=json.loads(line); key=case_key(r)
                anchor={scope:source_anchor(p) for scope,p in r['probability_endpoints'].items()}
                if key in refs and refs[key]!=anchor: raise ValueError('Parent source anchor depends on target')
                refs[key]=anchor
            unique={}
            for line in details.read_text().splitlines():
                r=json.loads(line); key=case_key(r)
                source=dict(positions=[dict(recipient=p['recipient'],donor=p['donor'],source_atoms=p['source_atoms']) for p in r['positions']],
                            scale=r['common_source_dose_scale'])
                if key in unique:
                    if unique[key][1]!=source: raise ValueError('Source export depends on target')
                else: unique[key]=(r,source)
            factors=np.load(checked(old['factors_path'],old['factors_sha256']),allow_pickle=False)
            findex={(int(s),int(a),int(t)):i for i,(s,a,t) in enumerate(zip(factors['source_seed'],factors['source_atom'],factors['target_seed']))}
            tm_path=checked(old['token_manifest_path'],old['token_manifest_sha256']); tm=json.loads(tm_path.read_text()); ti=tm['outputs']['calibration']
            tp=checked(tm_path.parent.parent/ti['path'],ti['sha256'])
            tokens=np.fromfile(tp,dtype='<u2').reshape(ti['sequences'],old['context_length'])
            for key,(r,source) in unique.items():
                b=np.asarray(factors['source_basis'][findex[r['source_seed'],r['source_atom'],r['target_seed']],:,0],dtype=np.float64)
                if any(not np.array_equal(b,factors['source_basis'][ix,:,0]) for k,ix in findex.items() if k[:2]==key[:2]):
                    raise ValueError('Source basis depends on target')
                delta=np.zeros((old['context_length'],len(b)))
                for p in source['positions']:
                    atoms=p['source_atoms']; scalar=sum(t['signed_contribution'] for t in atoms['terms'])
                    if abs(scalar-atoms['total'])>1e-10: raise ValueError('Source term sum mismatch')
                    pos=p['recipient']['position']
                    if int(tokens[r['sequence'],pos])!=p['recipient']['token_id']: raise ValueError('Recipient token mismatch')
                    delta[pos]=atoms['total']*b
                cases.append(dict(key=key, tokens=tokens[r['sequence']].copy(), delta=delta, source=source, anchor=refs[key], parent=parent['run']))
            factors.close()
        checks['frozen_cases']=len(cases)==cfg['expected_cases']
        if not checks['frozen_cases']: raise ValueError('Frozen case count changed')
        checked(Path(old['model_local_dir'])/'config.json'); checked('.aris/compute/local-r006b1-env-spec.json')
        write(run/'inputs.json',dict(inputs=inputs)); timers['preparation_seconds']=time.perf_counter()-started
        import torch
        import transformers
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        env.update(torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), sae_framework='not_applicable_saved_source_edits')
        load_start=time.perf_counter()
        model=transformers.AutoModelForCausalLM.from_pretrained(old['model_local_dir'],local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(cfg['device'])
        tokenizer=transformers.AutoTokenizer.from_pretrained(old['model_local_dir'],local_files_only=True)
        timers['load_seconds']=time.perf_counter()-load_start
        contract=HookPointContract(old['hook_module_path'],5,'resid_post',old['hook_hidden_size']); hook=model.get_submodule(old['hook_module_path'])
        numeric=time.perf_counter()
        def forward(batch,delta=None):
            nonlocal forwards
            def edit(m,i,out):
                h=extract_primary_hook_tensor(out,contract)
                return replace_primary_hook_tensor(out,h-delta,contract)
            handle=hook.register_forward_hook(edit) if delta is not None else None
            try:
                with torch.no_grad(): logits=model(batch,use_cache=False).logits[0].cpu().numpy()
            finally:
                if handle: handle.remove()
            forwards+=1; return logits
        for ci,c in enumerate(cases):
            batch=torch.tensor(c['tokens'][None].astype(np.int64),device=cfg['device'])
            baseline=forward(batch); source=forward(batch,torch.tensor(c['delta'][None],dtype=torch.float32,device=cfg['device']))
            ps=[p['recipient']['position'] for p in c['source']['positions']]
            observed=endpoint_positions(c['tokens'],ps); max_error=0.
            for scope,pos in observed.items():
                actual=probability_metrics(baseline,source,source,c['tokens'],pos)
                for field in ('source_nll_deltas','source_to_baseline_kl'):
                    max_error=max(max_error,float(np.max(np.abs(np.asarray(actual[field])-np.asarray(c['anchor'][scope][field])))))
            if max_error>cfg['anchor_absolute_tolerance']: raise ValueError(f'Source endpoint replay mismatch {ci}: {max_error}')
            pb,pnew,orders=probability_order(baseline[ps],source[ps],cfg['top_k'])
            arrays[f'case{ci}_baseline']=pb.astype(np.float32); arrays[f'case{ci}_source']=pnew.astype(np.float32)
            positions=[]
            for pi,(p,order) in enumerate(zip(c['source']['positions'],orders)):
                def token_rows(ids):
                    return [dict(token_id=int(j), token=tokenizer.decode([int(j)]), baseline_probability=float(pb[pi,j]), source_probability=float(pnew[pi,j]), probability_delta=float(pnew[pi,j]-pb[pi,j])) for j in ids]
                pos=p['recipient']['position']; gold=observed_next_id(c['tokens'],pos)
                positions.append(dict(**p, observed_next_token=token_rows([gold])[0] if gold is not None else None, total_variation=float(np.abs(pnew[pi]-pb[pi]).sum()/2),
                                      candidates={k:token_rows(v) for k,v in order.items()}))
            row=dict(case_index=ci,case_key=list(c['key']),parent=c['parent'],source_scale=c['source']['scale'],
                     anchor_max_absolute_error=max_error,positions=positions,array_positions=ps,
                     scope='source-only descriptive replay; target endpoints not selected or recomputed')
            rows.append(row)
            with (run/'metrics.raw.jsonl').open('a',encoding='utf-8') as sink: sink.write(json.dumps(row,sort_keys=True)+'\n')
            if time.perf_counter()-numeric>cfg['numeric_budget_seconds']: raise TimeoutError('Source replay numerical budget')
        timers['numeric_seconds']=time.perf_counter()-numeric
        save_start=time.perf_counter(); np.savez_compressed(run/'source_probabilities.npz',**arrays); timers['save_seconds']=time.perf_counter()-save_start
        checks.update(expected_forwards=forwards==2*cfg['expected_cases'],all_source_anchors=True,
                      probabilities_normalized=all(np.max(np.abs(a.sum(axis=1)-1))<1e-6 for a in arrays.values()))
    except Exception as exc:
        if 'numeric' in locals(): timers['numeric_seconds_until_exit']=time.perf_counter()-numeric
        error=f'{type(exc).__name__}: {exc}'; (run/'stderr.log').write_text(traceback.format_exc())
    env.update(python=sys.executable,python_version=platform.python_version(),platform=platform.platform(),numpy=np.__version__)
    write(run/'environment.json',env); write(run/'inputs.json',dict(inputs=inputs))
    status='PASS' if error is None and checks and all(checks.values()) else 'FAIL'
    summary=dict(status=status,error=error,checks=checks,model_forwards=forwards,rows=len(rows),timings=timers,wall_seconds=time.perf_counter()-started,
                 metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),generator_script_path='scripts/replay_f4_source_probability.py',generator_script_sha256=sha256(Path(__file__)))
    write(run/'metrics.summary.json',summary); write(run/'stdout.log',summary); write(run/'status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()))
    validation=validate_run_directory(run); write(run/'contract_validation.json',dict(ok=validation.ok,errors=list(validation.errors)))
    print(json.dumps(dict(summary=summary,contract_ok=validation.ok,contract_errors=list(validation.errors))),flush=True)
    return 0 if status=='PASS' and validation.ok else 1


if __name__=='__main__': raise SystemExit(main())
