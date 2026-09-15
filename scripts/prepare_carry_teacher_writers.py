"""Sparse decoder representations of a frozen functional writer.

Classical simultaneous greedy pursuit and SciPy NNLS are numerical components.
The saved paths diagnose write representation; generation tests their function.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import nnls
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def identity(p):
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def pursuit(decoder, targets, budgets, positive):
    norms = np.linalg.norm(targets, axis=0)
    unit = targets / norms
    residual = unit.copy()
    ids = []
    records = []
    snapshots = {}
    previous = float(np.square(residual).sum())
    for step in range(max(budgets)):
        correlation = decoder.T @ residual
        score = np.square(np.maximum(correlation, 0) if positive else correlation).sum(1)
        score[ids] = -np.inf
        j = int(np.argmax(score))
        if score[j] <= 1e-20:
            break
        ids.append(j)
        design = decoder[:, ids]
        coefficient = (np.stack([nnls(design, y, maxiter=10*len(ids))[0]
                                for y in unit.T], axis=1) if positive
                       else np.linalg.lstsq(design, unit, rcond=None)[0])
        residual = unit - design @ coefficient
        loss = float(np.square(residual).sum())
        assert loss <= previous + 1e-9, (loss, previous)
        previous = loss
        if step + 1 in budgets:
            physical = coefficient * norms
            gradient = design.T @ (design @ coefficient - unit)
            kkt = (max(float(np.maximum(-gradient, 0).max()),
                       float(np.abs(coefficient*gradient).max())) if positive
                   else float(np.abs(gradient).max()))
            assert kkt < 1e-6, kkt
            snapshots[step+1] = (np.array(ids), physical.T)
            records.append(dict(members=step+1, relative_residual=np.linalg.norm(residual,axis=0).tolist(),
                                coefficient_norm=np.linalg.norm(physical,axis=0).tolist(),
                                support_kkt_error=kkt))
            print(json.dumps(dict(positive=positive, **records[-1])), flush=True)
    return snapshots, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[64, 128])
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=False)
    started = stamp(); wall = time.perf_counter(); cpu = time.process_time()
    parent = ROOT/'runs/REFORM_R52_carry_readwrite_source_v1_20260915'
    checkpoint = Path('D:/CCAD_Storage/training_curves/REFORM_R32_qwen_l23_topk_five_seed_16m_v1_20260914/step_16384/topk_seed1.pt')
    cfg = json.loads((parent/'config.resolved.json').read_text())
    (out/'config.resolved.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'status.json').write_text(json.dumps(dict(status='RUNNING',started_at_utc=started)))
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        decoder = torch.load(checkpoint,map_location='cpu',weights_only=True)['decoder.weight'].double().numpy()
        decoder_norm = np.linalg.norm(decoder,axis=0)
        normalized = decoder / decoder_norm
        with np.load(parent/'rule_members_seed1.npz') as data:
            payload = {k:data[k] for k in data.files}
        raw = payload['readwrite_raw_writer'].astype(np.float64)
        records = []
        for positive in [False,True]:
            target = np.stack([raw[0],-raw[0],raw[1],-raw[1]]).T if positive else raw.T
            snapshots, path = pursuit(normalized,target,args.budgets,positive)
            records.append(dict(positive=positive,path=path))
            for k,(ids,weights) in snapshots.items():
                name = f'teacher_{"positive" if positive else "signed"}_{k}'
                payload[name]=ids
                payload[name+'_writer']=(weights/decoder_norm[ids]).astype(np.float32)
                payload[name+'_readout']=payload['readwrite_code_64_readout']
                payload[name+'_read_indices']=payload['readwrite_code_64']
                payload[name+'_operator']=np.array('cone' if positive else 'code')
                payload[name+'_nonnegative_update']=np.array(True)
        # A frozen code-reader/raw-writer reference shares the exact source reader.
        payload['code_read_raw_write']=payload['readwrite_code_64']
        payload['code_read_raw_write_readout']=payload['readwrite_code_64_readout']
        payload['code_read_raw_write_writer']=payload['readwrite_raw_writer']
        payload['code_read_raw_write_operator']=np.array('code_to_raw')
        payload['code_read_raw_write_nonnegative_update']=np.array(False)
    np.savez_compressed(out/'rule_members_seed1.npz',**payload)
    report=dict(started_at_utc=started,ended_at_utc=stamp(),wall_seconds=time.perf_counter()-wall,
                process_cpu_seconds=time.process_time()-cpu,paths=records,
                inputs=[identity(checkpoint),identity(parent/'rule_members_seed1.npz'),identity(Path(__file__))],
                budgets=args.budgets,read_members=64,source_seed=1,
                scope='Frozen functional teacher; no new task labels, output gradients or generation. Signed execution clips final codes; positive branches add codes selected by predicted query sign. Shared writing support across arities/signs; reading support separate.')
    (out/'TEACHER_WRITERS.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'status.json').write_text(json.dumps(dict(status='PASS',started_at_utc=started,ended_at_utc=report['ended_at_utc']),indent=2)+'\n')
    print(json.dumps(dict(output=str(out),wall_seconds=report['wall_seconds'])))


if __name__ == '__main__':
    main()
