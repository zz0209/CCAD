"""Check classifier-training sensitivity using frozen intervention representations."""
from pathlib import Path
import argparse
import json
import sys
import time
import traceback

from run_causalgym_multisite import MultisiteWork, write
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args()
    cfg=json.loads(a.config.read_text())
    work=MultisiteWork(cfg,a.config,['scripts/shift_cached_head_check.py',
        'scripts/run_causalgym_multisite.py','scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py'])
    error=None
    try:
        import torch
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.cuda.set_device(cfg['device'])
        torch.cuda.reset_peak_memory_stats()
        work.torch,work.device=torch,torch.device(cfg['device'])
        work.environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,
            device=torch.cuda.get_device_name(),cpu_threads=2)
        panel=json.loads(work.checked(cfg['panel']).read_text())
        train=[r for r in panel['rows'] if r['split']=='train']
        evaluation=[r for r in panel['rows'] if r['split']=='test']
        result={}
        replay_errors=[]
        for run_name in cfg['input_runs']:
            run=Path(run_name)
            status=json.loads(work.checked(run/'status.json').read_text())
            assert status['status']=='PASS',run
            original=json.loads(work.checked(run/'config.resolved.json').read_text())
            target=original['target_seed']
            membership=json.loads(work.checked(run/'task_membership.json').read_text())
            for method in original['methods']:
                if method in ['none','source'] and target!=cfg['seeds'][0]:
                    continue
                for query in original['queries']:
                    if method=='none' and query!='full':
                        continue
                    key=method+'__'+query
                    # Cache identities are recorded once; no model forward pass is used.
                    x=np.load(work.checked(run/(key+'__train.npy')))
                    y=np.load(work.checked(run/(key+'__evaluation.npy')))
                    assert x.shape==(len(train),512) and y.shape==(len(evaluation),512)
                    for task in original['tasks']:
                        name=task['name']
                        ti,ei=(membership[name][k] for k in ['train_indices','evaluation_indices'])
                        labels=np.array([int(train[i]['profession']==task['positive']) for i in ti])
                        yy=np.array([int(evaluation[i]['profession']==task['positive']) for i in ei])
                        gender=np.array([evaluation[i]['gender'] for i in ei])
                        xx=torch.tensor(x[ti],device=work.device)
                        tlabels=torch.tensor(labels,device=work.device,dtype=torch.float32)
                        logits=np.empty((len(cfg['probe_seeds']),len(cfg['epochs']),len(ei)),np.float32)
                        weights=np.empty((len(cfg['probe_seeds']),len(cfg['epochs']),513),np.float32)
                        for si,seed in enumerate(cfg['probe_seeds']):
                            torch.manual_seed(seed)
                            head=torch.nn.Linear(512,1,device=work.device)
                            optimizer=torch.optim.AdamW(head.parameters(),lr=original['probe_lr'])
                            for epoch in range(1,max(cfg['epochs'])+1):
                                for start in range(0,len(ti),original['probe_batch_size']):
                                    values=head(xx[start:start+original['probe_batch_size']]).squeeze(-1)
                                    loss=torch.nn.functional.binary_cross_entropy_with_logits(
                                        values,tlabels[start:start+original['probe_batch_size']])
                                    optimizer.zero_grad();loss.backward();optimizer.step()
                                if epoch not in cfg['epochs']:
                                    continue
                                e=cfg['epochs'].index(epoch)
                                weight=head.weight.detach().cpu().numpy()
                                bias=head.bias.detach().cpu().numpy()
                                value=(y[ei]@weight.T+bias).ravel()
                                logits[si,e]=value
                                weights[si,e]=np.r_[weight.ravel(),bias]
                                if seed==42 and epoch==1:
                                    saved=np.load(work.checked(run/f'{key}__{name}__probe42.npz'))
                                    difference=float(np.max(np.abs(value-saved['logits'])))
                                    assert np.array_equal(value>0,saved['logits']>0),(target,key,name)
                                    assert difference<2e-5,(target,key,name,difference)
                                    replay_errors.append(difference)
                                correct=(value>0)==yy
                                cells=[float(correct[(yy==c)&(gender==g)].mean()) for c in [0,1] for g in [0,1]]
                                work.record(kind='cached_head',task=name,component=name,row_id=epoch,mode='retrained',
                                    method=method,seed=seed,target_seed=target,operation=query,
                                    accuracy=float(correct.mean()),worst_group=min(cells),group_accuracy=cells)
                        filename=f'target{target}__{key}__{name}.npz'
                        np.savez_compressed(work.run/filename,logits=logits,weights=weights,
                            labels=yy,gender=gender,document_sha256=[evaluation[i]['document_sha256'] for i in ei])
                        result[f'{target}/{key}/{name}']=filename
                        del xx,tlabels
                    work.progress('CACHED_HEADS',target_seed=target,method=method,query=query,completed=len(result))
                    if time.perf_counter()-work.wall_start>cfg['budget_seconds']:
                        raise TimeoutError('Bounded cached-head sensitivity check')
            del x,y
        work.checks.update(frozen_representations=True,no_new_model_forwards=work.sequence_forwards==0,
            seed42_epoch1_replay=True)
        write(work.run/'HEAD_CHECK_INDEX.json',dict(files=result,probe_seeds=cfg['probe_seeds'],
            epochs=cfg['epochs'],primary_replay_max_logit_error=max(replay_errors),
            replay_cells=len(replay_errors),scope=cfg['scope']))
    except Exception:
        error=traceback.format_exc()
    return work.finish(error)


if __name__=='__main__':
    raise SystemExit(main())
