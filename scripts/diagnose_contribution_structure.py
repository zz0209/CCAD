"""Diagnose source spectrum and common-direction explanations without refitting.

PC projections are a decomposition of frozen vector errors, not new edits.
The global-field projection is in L2(inputs; hook vectors), not a pointwise PC.
"""
import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from run_contribution_completion import ROOT, np, entry, aggregate, write, sha256, validate_run_directory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    cfg = json.loads(ap.parse_args().config.read_text())
    run = ROOT/'runs'/cfg['run_id']
    run.mkdir(exist_ok=False)
    start = time.perf_counter()
    write(run/'config.resolved.json', cfg)
    files = []
    for rel in ['scripts/diagnose_contribution_structure.py', 'scripts/run_contribution_completion.py',
                'scripts/run_f4_source_reference_causal.py', 'scripts/run_r011s1_raw_hook_asset.py', 'src/ccad/artifacts.py']:
        p = ROOT/rel
        dst = run/'source_snapshot'/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
        files.append(dict(path=rel, sha256=sha256(p), bytes=p.stat().st_size, snapshot_path='source_snapshot/'+rel))
    write(run/'code_hashes.json', dict(files=files, aggregate_sha256=aggregate(files), snapshot_root='source_snapshot'))
    write(run/'manifest.json', dict(schema_version='contribution.structure.v1', run_id=cfg['run_id'],
          run_parent=cfg['grain_run'], purpose=cfg['purpose'], milestone='resolve_core_confound',
          evidence_level='posthoc_development_diagnostic', started_utc=datetime.now(timezone.utc).isoformat(),
          project_root=str(ROOT), config_hash=sha256(run/'config.resolved.json'), code_snapshot_hash=aggregate(files),
          source_snapshot_required=True, audit_opened=False, candidate_family_frozen=True,
          mean_constants_source_split='parent independent mean; covariance centering for donor differences',
          threshold_source_split='raw PCA discovery; all fixed cutpoints, no outcome selection',
          statistics_unit='overlapping source groups, shared documents and five seeds', device='cuda:0',
          seeds=[1,2,3,4,5], resource_lease='gpu-0', resource_lease_reason=cfg['budget']))
    for f in ['stdout.log', 'stderr.log', 'metrics.raw.jsonl']:
        (run/f).touch()
    write(run/'status.json', dict(status='RUNNING'))
    inputs, rows, source_rows, natural_rows = [], [], [], []
    checks, env, error = {}, {}, None

    def checked(path):
        p = Path(path)
        p = p if p.is_absolute() else ROOT/p
        inputs.append(entry(p, 'Existing CCAD development artifact', 'input'))
        return p

    def load(path):
        return json.loads(checked(path).read_text())

    def progress(stage, **kw):
        v = dict(stage=stage, seconds=time.perf_counter()-start, **kw)
        write(run/'progress.json', v)
        print(json.dumps(v), flush=True)
        if v['seconds'] > cfg['budget_seconds']:
            raise TimeoutError('Bounded diagnostic budget exceeded')

    try:
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()

        def tensor(x):
            return torch.tensor(x, dtype=torch.float64, device='cuda:0')

        parent, grain, natural = [ROOT/'runs'/cfg[k] for k in ['parent_run','grain_run','natural_run']]
        for p in [parent, grain, natural]:
            assert load(p/'status.json')['status'] == 'PASS'
        pcfg = load(parent/'config.resolved.json')
        groups = load(grain/'source_groups.json')['groups']
        rawspec = next(x for x in load(pcfg['raw_manifest'])['splits'] if x['split']=='discovery')
        with np.load(checked(parent/'codes_s1_discovery.npz')) as ar:
            row_indices = ar['row_indices']
        rawmap = np.memmap(rawspec['path'], dtype='<f4', mode='r', shape=tuple(rawspec['shape']))
        raw = np.array(rawmap[row_indices], dtype=np.float64)
        np.save(run/'raw_discovery_subset.npy', raw)
        inputs.append(entry(run/'raw_discovery_subset.npy', 'Parent raw manifest indexed subset', 'materialized_input'))
        write(run/'raw_subset_provenance.json', dict(parent=rawspec, row_indices=row_indices.tolist(),
              actual_subset_sha256=sha256(run/'raw_discovery_subset.npy'), full_raw_rehash=False))
        x = tensor(raw-raw.mean(0))
        eig, u = torch.linalg.eigh(x.T@x/len(x))
        eig, u = eig.flip(0), u.flip(1)
        u = u[:, :max(cfg['pc_ranks'])]
        np.savez_compressed(run/'raw_pca.npz', eigenvalues=eig.cpu().numpy(), components=u.cpu().numpy(), row_indices=row_indices)
        checks['pca_orthonormal'] = float(torch.max(torch.abs(u.T@u-torch.eye(u.shape[1], device='cuda:0')))) < 1e-10
        del x, raw
        z, cov, dec, mask, kernels, sg = {}, {}, {}, {}, {}, {}
        with np.load(checked(parent/'decoders_means.npz')) as ar:
            for s in range(1,6):
                dec[s] = tensor(ar[f'decoder_{s}'])
        for s in range(1,6):
            qq = groups[str(s)]
            a = torch.zeros((len(dec[s]),len(qq)),dtype=torch.float64,device='cuda:0')
            for j,q in enumerate(qq):
                a[q['source_atoms'],j] = 1
            mask[s] = a
            for split in ['discovery','calibration']:
                with np.load(checked(parent/f'codes_s{s}_{split}.npz')) as ar:
                    v = tensor(ar['codes'])
                v -= v.mean(0)
                c = v.T@v/len(v)
                for j,q in enumerate(qq):
                    ids = q['source_atoms']
                    ci, di = c[ids][:,ids], dec[s][ids]
                    ec, ev = torch.linalg.eigh(ci)
                    csqrt = (ev*torch.sqrt(torch.clamp(ec,min=0)))@ev.T
                    spectrum = torch.clamp(torch.linalg.eigvalsh(csqrt@(di@di.T)@csqrt), min=0).flip(0)
                    energy = float(spectrum.sum())
                    dpc = di@u
                    pc_energy = torch.sum(dpc*(ci@dpc),dim=0).cumsum(0)
                    sr = dict(**q, group_index=j, split=split, energy=energy,
                         effective_rank=energy**2/float(torch.sum(spectrum*spectrum)),
                         rank_one_share=float(spectrum[0])/energy, spectrum=spectrum.cpu().numpy().tolist(),
                         raw_pc_shares={str(k):float(pc_energy[k-1])/energy for k in cfg['pc_ranks']})
                    source_rows.append(sr)
                if split=='calibration':
                    z[s], cov[s] = v,c
                    for k in [0]+cfg['pc_ranks']:
                        d = dec[s] if k==0 else dec[s]@u[:,:k]
                        g = c*(d@d.T)
                        kernels[s,k] = g
                        sg[s,k] = torch.sum(a*(g@a),dim=0)
            progress('source_spectrum', source=s, groups=len(source_rows))
        write(run/'source_structure.json', dict(rows=source_rows, raw_explained_variance={str(k):float(eig[:k].sum()/eig.sum()) for k in cfg['pc_ranks']}))
        old = {(r['source_seed'],r['target_seed'],r['query_id'],r['size'],r['kind']):r for r in
               [json.loads(v) for v in checked(grain/'metrics.raw.jsonl').read_text().splitlines()]}
        replay_error, min_complement = 0., 1.
        for s in range(1,6):
            source_global_norm = kernels[s,0].sum()
            source_global_inner = torch.sum(kernels[s,0]@mask[s],dim=0)
            for t in range(1,6):
                if s==t:
                    continue
                with np.load(checked(grain/f'maps_s{s}_t{t}.npz')) as ar:
                    b = tensor(ar[cfg['method']])
                cross = z[t].T@z[s]/len(z[s])
                totals, parts = None, {}
                for k in [0]+cfg['pc_ranks']:
                    ds, dt = (dec[s],dec[t]) if k==0 else (dec[s]@u[:,:k],dec[t]@u[:,:k])
                    h = cross*(dt@ds.T)
                    err = sg[s,k]-2*torch.sum(b*(h@mask[s]),dim=0)+torch.sum(b*(kernels[t,k]@b),dim=0)
                    if k==0:
                        totals = err
                        pred_global_inner = b.T@torch.sum(h,dim=1)
                        eglobal = (pred_global_inner-source_global_inner)**2/source_global_norm
                        yglobal = source_global_inner**2/source_global_norm
                    else:
                        parts[k] = err
                for j,q in enumerate(groups[str(s)]):
                    total, energy = float(totals[j]), float(sg[s,0][j])
                    expected = old[s,t,q['query_id'],q['size'],q['kind']]['methods'][cfg['method']]['absolute_error']
                    replay_error = max(replay_error,abs(total-expected))
                    pr = {}
                    for k in cfg['pc_ranks']:
                        pe, py = float(parts[k][j]),float(sg[s,k][j])
                        ce, cy = total-pe,energy-py
                        min_complement = min(min_complement,ce,cy)
                        pr[str(k)] = dict(source_pc_share=py/energy, pc_error=pe, complement_error=ce,
                             pc_relative_error=pe/py if py>1e-20 else None,
                             complement_relative_error=ce/cy if cy>1e-20 else None)
                    rows.append(dict(**q,target_seed=t,method=cfg['method'],source_energy=energy,error=total,
                        relative_error=total/energy,components=pr,global_field_source_share=float(yglobal[j])/energy,
                        global_field_complement_relative_error=float((totals[j]-eglobal[j])/(sg[s,0][j]-yglobal[j]))))
                progress('frozen_map_decomposition',source=s,target=t,rows=len(rows))
        checks['frozen_error_replays_parent'] = replay_error < 1e-7
        checks['nonnegative_orthogonal_parts'] = min_complement > -1e-7
        # Natural operations are already materialized vectors, no additional model call.
        plans = load(natural/'operation_plan.json')['plans']
        edits = np.load(checked(natural/'edits.npy'))
        un = u.cpu().numpy()
        for p in plans:
            y = edits[p['source_edit_index']]
            for m in p['methods']:
                if m['method']!=cfg['method']:
                    continue
                residual = edits[m['edit_index']]-y
                energy, err = float(y@y),float(residual@residual)
                parts = {}
                for k in cfg['pc_ranks']:
                    yp, ep = y@un[:,:k],residual@un[:,:k]
                    parts[str(k)] = dict(source_pc_energy=float(yp@yp),error_pc=float(ep@ep),
                        source_complement_energy=energy-float(yp@yp),error_complement=err-float(ep@ep))
                natural_rows.append(dict(case_id=p['case_id'],query_id=p['query_id'],source_seed=p['source_seed'],
                    target_seed=m['target_seed'],size=p['size'],kind=p['kind'],source_energy=energy,error=err,components=parts))
        write(run/'natural_decomposition.json',dict(rows=natural_rows,scope='Existing exposed natural operations, no refit or new LM call'))
        checks.update(all_source_groups=len(source_rows)==2240,all_cross_comparisons=len(rows)==4480,all_natural=len(natural_rows)==960,
                      zero_new_forward=True,all_original_queries_retained=True)
        env = dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                   cpu_threads=torch.get_num_threads(),gpu=torch.cuda.get_device_name(),peak_vram_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        (run/'stderr.log').write_text(traceback.format_exc())
    with (run/'metrics.raw.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row)+'\n')
    status = 'PASS' if error is None and checks and all(checks.values()) else 'FAIL'
    summary = dict(status=status,error=error,checks=checks,rows=len(rows),source_rows=len(source_rows),natural_rows=len(natural_rows),
                   wall_seconds=time.perf_counter()-start,scope=cfg['scope'],metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'),
                   generator_script_path='scripts/diagnose_contribution_structure.py',generator_script_sha256=files[0]['sha256'])
    for name,obj in [('environment.json',env),('inputs.json',dict(inputs=inputs)),('metrics.summary.json',summary),('stdout.log',summary),
                     ('status.json',dict(status=status,error=error,updated_utc=datetime.now(timezone.utc).isoformat()))]:
        write(run/name,obj)
    v = validate_run_directory(run)
    write(run/'contract_validation.json',dict(ok=v.ok,errors=list(v.errors)))
    print(json.dumps(dict(summary=summary,contract_ok=v.ok,errors=list(v.errors))),flush=True)
    return 0 if status=='PASS' and v.ok else 1


if __name__=='__main__':
    raise SystemExit(main())
