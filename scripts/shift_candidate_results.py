"""Compare fixed-budget candidate retrieval on the exposed SHIFT development panel."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import numpy as np
import shift_confirmation_results as metrics
from shift_query_results import ROOT, ART, read, identity, load_logits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='REFORM_R60_shift_context_candidates_dev_v1_20260915')
    parser.add_argument('--output',default='R60_CANDIDATE_ANALYSIS.json')
    parser.add_argument('--export-paper',action='store_true')
    args=parser.parse_args();run=ROOT/'runs'/args.run
    old=ROOT/'runs/REFORM_R59_shift_query_granularity_v1_20260915'
    for folder in [run,old]:
        assert read(folder/'status.json')['status']=='PASS'
        assert read(folder/'contract_validation.json')['ok']
    cfg=read(run/'config.resolved.json');prior_cfg=read(old/'config.resolved.json')
    matched=['source_manifest','source_parameters','notebook','frozen_source_run',
             'model_local_dir','target_directory','target_seed','paired_tokens',
             'fit_sequences','fit_batch_sequences','fit_iterations','members_per_source',
             'ridge','functional_weight','context_response','eval_batch_size','queries']
    assert all(cfg[k]==prior_cfg[k] for k in matched)
    source=Path(cfg['frozen_source_run'])
    panel=[row for row in read(source/'panel.json')['rows'] if row['split']=='dev']
    order={row['document_sha256']:i for i,row in enumerate(panel)}
    assert len(order)==len(panel)==695
    values=load_logits(run,order);prior=load_logits(old,order)
    source_error={q:float(np.max(np.abs(values['source/'+q]-prior['source/'+q]))) for q in cfg['queries']}
    assert max(source_error.values())==0.,source_error
    base=np.load(source/'dev_none.npz')['logits']
    values['none/full']=base;prior['none/full']=base
    labels=np.array([r['label'] for r in panel]);gender=np.array([r['gender'] for r in panel])
    strata=[np.flatnonzero((labels==y)&(gender==g)) for y in [0,1] for g in [0,1]]
    rng=np.random.default_rng(60015);counts=np.zeros((2000,len(panel)))
    for ix in strata:
        for draw,sample in enumerate(rng.choice(ix,size=(len(counts),len(ix)),replace=True)):
            counts[draw]+=np.bincount(sample,minlength=len(panel))
    metrics.METHODS=cfg['methods']
    result,new_boot=metrics.summarize(values,labels,strata,counts)
    metrics.METHODS=['source','geometry','native','raw']
    original,old_boot=metrics.summarize(prior,labels,strata,counts)
    contrasts={}
    for method in ['native','raw','geometry']:
        for family in metrics.FAMILIES:
            key=method+'/'+family;contrasts[key]={}
            for metric in metrics.METRICS:
                difference=result['families'][key][metric]['value']-original['families'][key][metric]['value']
                contrasts[key][metric]=dict(difference=difference,
                    document_ci=metrics.interval(new_boot[key][metric]-old_boot[key][metric]))
    fit=read(run/'RELATION_FIT.json');old_fit=read(old/'RELATION_FIT.json')
    assert all(fit[site]['candidates']==old_fit[site]['candidates'] and
               fit[site]['selected']==old_fit[site]['selected'] for site in fit)
    retrieval=read(run/'CANDIDATE_RETRIEVAL.json')
    support={site:dict(candidate_count=v['candidates'],target_allowance=v['selected'],
                introduced_candidates=len(retrieval[site]['outside_geometry']),
                selected_outside_geometry=v['selected_outside_geometry'],
                active_source_distributions=sum(r['status']=='MATCHED' for r in retrieval[site]['rows']),
                source_members=v['source_members'],retrieval_seconds=retrieval[site]['retrieval_seconds'])
             for site,v in fit.items()}
    records=[identity(p) for p in [run/'config.resolved.json',run/'metrics.raw.jsonl',
             run/'RELATION_FIT.json',run/'CANDIDATE_RETRIEVAL.json',old/'config.resolved.json',
             old/'metrics.raw.jsonl',source/'panel.json',source/'dev_none.npz',Path(__file__),
             Path(metrics.__file__)]]
    out=dict(written_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
             run=args.run,original_run=old.name,result=result,original=original,
             new_minus_original=contrasts,support=support,source_max_logit_error=source_error,sources=records,
             scope='One exposed development target and 695 documents; 2000 paired profession/gender-stratified document resamples, seed60015. Same source, unique candidate counts and final allowance. Context retrieval changes the pool for all within-run methods. Comparisons to original methods use their original geometry pool. No independent confirmation or cross-seed inference.')
    (ART/args.output).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    if args.export_paper:
        (ROOT/'paper/data/human_candidate_retrieval.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
        lines=[r'\begin{tabular}{llrrrr}\toprule',
               r' & & \multicolumn{2}{c}{Source agreement} & \multicolumn{2}{c}{Part task outcomes}\\',
               r'Executor & Candidate pool & Parts & Unions & Profession & Worst group\\\midrule']
        for method,label in [('geometry','Geometry'),('native','Member relation'),('raw','Source-dir. readout')]:
            for cohort,summary in [('Geometric',original),('Context hybrid',result)]:
                v=summary['families'];numbers=[v[method+'/'+f]['balanced_agreement']['value'] for f in ['parts','unions']]
                numbers += [v[method+'/parts'][m]['value'] for m in ['profession','worst_group']]
                lines.append(label+' & '+cohort+' & '+' & '.join(f'{100*n:.2f}' for n in numbers)+r'\\')
        lines.append(r'\bottomrule\end{tabular}')
        (ROOT/'paper/tables/human_candidate_retrieval.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for method in cfg['methods']:
        print(method,{f:round(100*result['families'][method+'/'+f]['balanced_agreement']['value'],3) for f in metrics.FAMILIES})
    print('native context-minus-original',contrasts['native/parts']['balanced_agreement'])
    print('selected outside geometry',sum(len(v['selected_outside_geometry']) for v in support.values()))


if __name__=='__main__':main()
