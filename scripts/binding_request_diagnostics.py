"""Extract source-confidence and target-realization diagnostics from saved runs."""
from pathlib import Path
import argparse,json,sys,traceback
from run_causalgym_multisite import MultisiteWork,ROOT,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);args=p.parse_args()
    cfg=json.loads(args.config.read_text());w=MultisiteWork(cfg,args.config,
        ['scripts/binding_request_diagnostics.py','scripts/run_causalgym_multisite.py',
         'scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py'])
    error=None
    try:
        import torch,numpy as np
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')
        w.torch=torch;w.device=torch.device(cfg['device']);torch.cuda.set_device(w.device);torch.cuda.reset_peak_memory_stats()
        parent=ROOT/cfg['parent_run'];assert json.loads(w.checked(parent/'status.json').read_text())['status']=='PASS'
        pc=json.loads(w.checked(parent/'config.resolved.json').read_text());panel=json.loads(w.checked(parent/'panel.json').read_text());rows=panel['rows']
        raw=[json.loads(s) for s in w.checked(parent/'metrics.raw.jsonl').read_text().splitlines()]
        lookup={(r['row_id'],r['seed'],r['method'],r['operation']):r for r in raw if r['kind']=='intervention'}
        with np.load(w.checked(parent/'states.npz')) as saved:h=torch.tensor(saved['hidden_13'],device=w.device)
        donor=torch.tensor([r['paired_row'] for r in rows],device=w.device);dh=h[donor]-h
        fit=[r['row_id'] for r in rows if r['split']=='fit' and not r['donor']]
        evaluation=[r['row_id'] for r in rows if r['split']!='fit' and not r['donor']]
        tr=ROOT/pc['training_runs']['13'];tc=json.loads(w.checked(tr/'config.resolved.json').read_text());snaps=json.loads(w.checked(tr/'checkpoints.json').read_text())['checkpoints']
        sys.path.extend([tc['dictionary_source_dir'],tc['dictionary_overlay_dir']]);from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.checked(Path(tc['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py');w.checked(Path(tc['dictionary_source_dir'])/'LICENSE')
        def load(seed):
            snap=next(x for x in snaps if x['seed']==seed and x['step']==pc['checkpoint_step'] and x['objective']=='topk')
            ae=AutoEncoderTopK(h.shape[-1],tc['dict_size'],tc['k']).to(w.device)
            ae.load_state_dict(torch.load(w.checked(snap['path']),map_location=w.device,weights_only=True));ae.eval();ae.requires_grad_(False)
            with torch.no_grad():z=ae.encode(h.flatten(0,1)).reshape(len(h),2,-1)
            return ae,z
        for seed in cfg['seeds']:
            target_seed=seed%5+1;source,zs=load(seed);target,zt=load(target_seed);ds=source.decoder.weight.T;dt=target.decoder.weight.T
            z=np.load(w.checked(parent/f'binding_relation_s{seed}_t{target_seed}.npz'))
            def tensor(k):return torch.tensor(z[k],device=w.device)
            si=tensor('source_indices');ti=tensor('target_indices');q=tensor('source_query');dx=zt[donor]-zt
            truth=((zs[donor]-zs)[:,:,si]*q)@ds[si]
            predicted=(dx[:,:,ti]@tensor('code_readout')*q)@ds[si]
            fields={'union_member_country':(dx[:,:,ti]*tensor('union_member_country'))@dt[ti]}
            fields['native_scalar_country']=fields['union_member_country']*tensor('native_scalar_alpha')[...,None]
            for name in ['synthesized_code_bank','synthesized_code_shared_support']:
                c=torch.zeros_like(zt);c.scatter_(-1,tensor(name+'_indices'),tensor(name+'_coefficients'));fields[name]=c@dt
            ac=tensor('assignment');fields['assignment_country']=(dx[:,:,ac]*q)@dt[ac]
            countries=sorted({v for r in rows for v in r['countries']});means=[]
            for country in countries:
                selected=[zt[r['row_id'],j] for r in rows if r['split']=='fit' for j in range(2) if r['countries'][j]==country]
                means.append(torch.stack(selected).mean(0))
            means=torch.stack(means);direct=torch.zeros_like(zt)
            for r in rows:
                other=rows[r['paired_row']]
                for j in range(2):
                    u,v=[countries.index(x['countries'][j]) for x in [r,other]]
                    keep=torch.argsort((means[v]-means[u]).abs()*dt.norm(dim=1),descending=True,stable=True)[:64]
                    direct[r['row_id'],j,keep]=1
            fields['target_country']=(dx*direct)@dt
            for operation,sites in [('first',[0]),('second',[1]),('both',[0,1])]:
                denom=predicted[:,sites].square().sum((1,2)).clamp_min(1e-12)
                for method,field in fields.items():
                    squared=(field[:,sites]-predicted[:,sites]).square().sum((1,2))
                    residual=(squared/denom).cpu().numpy()
                    truth_error=((field[:,sites]-truth[:,sites]).square().sum((1,2))/truth[:,sites].square().sum((1,2)).clamp_min(1e-12)).cpu().numpy()
                    cosine=((field[:,sites]*predicted[:,sites]).sum((1,2))/(field[:,sites].square().sum((1,2))*denom).sqrt().clamp_min(1e-12)).cpu().numpy()
                    normratio=(field[:,sites].square().sum((1,2))/denom).sqrt().cpu().numpy()
                    for i in evaluation:
                        r=lookup[i,seed,method,operation];s=lookup[i,seed,'country64',operation];cr=lookup[i,seed,'code_readout_country',operation]
                        w.record(kind='request_diagnostic',task=r['task'],row_id=i,component=r['component'],mode='layer13',method=method,seed=seed,target_seed=target_seed,operation=operation,query=r['query'],split='exposed_development',
                            source_log_probability=s['expected_log_probability'],source_correct=s['correct'],code_log_probability=cr['expected_log_probability'],code_correct=cr['correct'],target_correct=r['correct'],
                            exact_source_answer=r['answer_id']==s['answer_id'],relative_prediction_residual=float(residual[i]),relative_true_source_residual=float(truth_error[i]),cosine=float(cosine[i]),norm_ratio=float(normratio[i]),edit_norm=r['edit_norm'])
            w.progress('DIAGNOSTICS',source_seed=seed,target_seed=target_seed)
        w.environment=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),precision='float32 matmul high',language_model_forwards=0)
        w.checks['no_new_language_model_outputs']=w.sequence_forwards==0
    except Exception:error=traceback.format_exc();print(error,file=sys.stderr)
    return w.finish(error)


if __name__=='__main__':raise SystemExit(main())
