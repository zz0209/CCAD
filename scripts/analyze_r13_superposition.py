"""Aggregate retained toy operation errors without treating directions as IID."""
from __future__ import annotations
import argparse,collections,datetime as dt,hashlib,json
from pathlib import Path

NUMBERS=['samples','operation_error','operation_energy','truth_error','truth_energy','function_error','function_energy','truth_function_error','truth_function_energy']
RATIOS={'operation_nmse':('operation_error','operation_energy'),'function_nmse':('function_error','function_energy'),'truth_nmse':('truth_error','truth_energy'),'truth_function_nmse':('truth_function_error','truth_function_energy')}
COMPARATORS=['ols_code','ols_code_clip','ols_affine_clip','ols_preactivation','dense_rectified','atom1_rectified','full_rectified','full_preactivation','target_full_reencode']

def sums(rows):
    result={k:sum(r[k] for r in rows) for k in NUMBERS}
    for key,(n,d) in RATIOS.items():result[key]=result[n]/result[d] if result[d] else None
    return result

def analyze(run):
    cfg=json.loads((run/'config.resolved.json').read_text());groups={};raw_count=0;method_counts=collections.Counter()
    with (run/'metrics.raw.jsonl').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line);raw_count+=1;method_counts[r['method']]+=1
            for family in ['all','pair01' if r['factor']<2 else 'other10']:
                key=tuple(r[k] for k in ['regime','toy_seed','source_seed','target_seed','split','method'])+(family,)
                item=groups.setdefault(key,{**{k:r[k] for k in ['regime','toy_seed','source_seed','target_seed','split','method']},'family':family,'rows':0,'member_sum':0,**{k:0 for k in NUMBERS}})
                item['rows']+=1;item['member_sum']+=r['target_members']
                for k in NUMBERS:item[k]+=r[k]
    rows=list(groups.values());pooled=[];comparisons=[];source_quality=[]
    methods=sorted(method_counts);seeds=cfg['sae_seeds']
    for regime in cfg['regimes']:
        for split in ['calibration','evaluation','shift']:
            for family in ['all','pair01','other10']:
                subset=[r for r in rows if (r['regime'],r['split'],r['family'])==(regime,split,family)]
                for method in methods:
                    vals=[r for r in subset if r['method']==method]
                    if not vals:continue
                    pooled.append(dict(regime=regime,split=split,family=family,method=method,mean_target_members=sum(r['member_sum'] for r in vals)/sum(r['rows'] for r in vals),**sums(vals)))
                ours=[r for r in subset if r['method']=='ols_rectified']
                for comparator in COMPARATORS:
                    other=[r for r in subset if r['method']==comparator]
                    if not ours or not other:continue
                    comparisons.append(dict(regime=regime,split=split,family=family,comparator=comparator,
                        function_relative_reduction=1-sums(ours)['function_error']/sums(other)['function_error'],
                        operation_relative_reduction=1-sums(ours)['operation_error']/sums(other)['operation_error'],
                        by_toy_seed=[dict(toy_seed=base,function_relative_reduction=1-sums([r for r in ours if r['toy_seed']==base])['function_error']/sums([r for r in other if r['toy_seed']==base])['function_error']) for base in cfg['toy_seeds']],
                        incident_seed_deletions=[dict(removed_seed=seed,function_relative_reduction=1-sums([r for r in ours if r['source_seed']!=seed and r['target_seed']!=seed])['function_error']/sums([r for r in other if r['source_seed']!=seed and r['target_seed']!=seed])['function_error']) for seed in seeds] if len(seeds)>2 else [],
                        per_direction=[dict(toy_seed=r['toy_seed'],source_seed=r['source_seed'],target_seed=r['target_seed'],function_relative_reduction=1-r['function_error']/next(v['function_error'] for v in other if (v['toy_seed'],v['source_seed'],v['target_seed'])==(r['toy_seed'],r['source_seed'],r['target_seed']))) for r in ours]))
                if family=='all':
                    for base in cfg['toy_seeds']:
                        for seed in seeds:
                            vals=[r for r in subset if r['method']=='source_encoder_oracle' and r['toy_seed']==base and r['source_seed']==seed]
                            source_quality.append(dict(regime=regime,split=split,toy_seed=base,sae_seed=seed,**sums(vals)))
    expected_factors=cfg['features'];coverage=[];optimization=[]
    for p in sorted(run.glob('*_fits.json')):
        data=json.loads(p.read_text());fits=data['fits'];by_pair=collections.defaultdict(list)
        for fit in fits:
            by_pair[(fit['source_seed'],fit.get('target_seed'))].append(fit['factor'])
            for method,model in fit.get('models',{}).items():
                if 'optimization' in model:optimization.append(dict(regime=fit['regime'],toy_seed=fit['toy_seed'],source_seed=fit['source_seed'],target_seed=fit['target_seed'],factor=fit['factor'],method=method,**model['optimization']))
        coverage.append(dict(file=p.name,maps=len(fits),all_factor_sets_complete=all(sorted(v)==list(range(expected_factors)) for v in by_pair.values()),empty_groups=sum(not fit['source_group'] for fit in fits)))
    return dict(run_id=run.name,written_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),config=cfg,metrics_raw_sha256=hashlib.sha256((run/'metrics.raw.jsonl').read_bytes()).hexdigest(),raw_rows=raw_count,method_counts=dict(method_counts),coverage=coverage,pooled=pooled,comparisons=comparisons,source_quality=source_quality,by_direction=rows,optimization=optimization,
                scope='Sums over every retained factor and generated block; repeated directions share source/target SAE and trained toy. Two toy seeds and incident-seed deletions are sensitivity results, not IID direction confidence intervals.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    result=analyze(args.run);args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    for regime in result['config']['regimes']:
        for split in ['evaluation','shift']:
            vals=[r for r in result['pooled'] if r['regime']==regime and r['split']==split and r['family']=='all']
            print(regime,split,{r['method']:round(r['function_nmse'],5) for r in vals if r['method'] in ['ols_affine_clip','ols_rectified','dense_rectified','full_rectified']})
    print(json.dumps(dict(out=str(args.out),rows=result['raw_rows'],coverage=result['coverage'])))
if __name__=='__main__':main()
