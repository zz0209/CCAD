"""Common-teacher results with lexical blocks and shared seeds kept explicit."""
import argparse,csv,json,statistics
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    raw=[json.loads(s) for s in (args.run/'metrics.raw.jsonl').read_text().splitlines()];source={};noop={};groups={};curve={}
    for r in raw:
        if r.get('checkpoint')=='transfer':
            key=(r['source_seed'],r['factor'],r['pair_id'])
            if r['method']=='source_native_teacher':source[key]=r
            elif r['method']=='noop_teacher_reference':noop[key]=r
        else:
            key=(r['checkpoint'],r['factor'],r['method'],r['split']);curve.setdefault(key,[]).append(r)
    for r in raw:
        if r.get('checkpoint')!='transfer' or r['source_seed']==r['target_seed']:continue
        key=(r['source_seed'],r['target_seed'],r['factor'],r['method'],r['split']);groups.setdefault(key,[]).append(r)
    summaries=[];blockrows=[]
    def aggregate(key,rows):
        base=dict(zip(['source_seed','target_seed','factor','method','split'],key));energy=error=teacher_kl=noop_kl=marginerr=0.;same=0;teacher_correct=0
        for r in rows:
            sk=(r['source_seed'],r['factor'],r['pair_id']);s=source[sk];n=noop[sk];e=s['delta_norm']**2;energy+=e;error+=r['teacher_vector_sqerror']*max(e,1e-12);teacher_kl+=r['kl_to_reference'];noop_kl+=n['kl_to_reference'];marginerr+=abs(r['plural_margin']-s['plural_margin']);same+=(r['plural_margin']>0)==(s['plural_margin']>0);teacher_correct+=s['donor_label_correct']
        return dict(**base,n=len(rows),lexical_blocks=len({r['block'] for r in rows}),target_members=rows[0].get('target_members'),prediction_rank_discovery=rows[0].get('prediction_rank_discovery'),correct=sum(r['donor_label_correct'] for r in rows),teacher_correct=teacher_correct,teacher_label_agreement=same/len(rows),mean_teacher_kl=teacher_kl/len(rows),mean_noop_kl=noop_kl/len(rows),kl_ratio=teacher_kl/noop_kl if noop_kl else None,vector_error_ratio=error/energy if energy else None,mean_margin_error=marginerr/len(rows))
    for key,rows in groups.items():
        summaries.append(aggregate(key,rows))
        for block in sorted({r['block'] for r in rows}):blockrows.append(dict(**aggregate(key,[r for r in rows if r['block']==block]),block=block))
    curves=[]
    for key,rows in curve.items():
        curves.append(dict(zip(['checkpoint','factor','method','split'],key),n=len(rows),lexical_blocks=len({r['block'] for r in rows}),correct=sum(r['donor_label_correct'] for r in rows),mean_vector_error=statistics.mean(r['relative_vector_sqerror'] for r in rows),mean_raw_reference_kl=statistics.mean(r['kl_to_reference'] for r in rows if r['kl_to_reference'] is not None) if any(r['kl_to_reference'] is not None for r in rows) else None))
    macro=[]
    for factor in ['subject','distractor']:
        methods=sorted({r['method'] for r in summaries})
        for method in methods:
            rows=[r for r in summaries if r['factor']==factor and r['method']==method and r['split']=='held_lexical_development'];kl=statistics.mean(r['mean_teacher_kl'] for r in rows);nk=statistics.mean(r['mean_noop_kl'] for r in rows)
            # Five leave-one-seed-out descriptive means, not independent folds
            # or a confidence interval. Removing a seed removes both edge roles.
            leave=[]
            for seed in range(1,6):
                kept=[r for r in rows if seed not in [r['source_seed'],r['target_seed']]]
                leave.append(statistics.mean(r['mean_teacher_kl'] for r in kept)/statistics.mean(r['mean_noop_kl'] for r in kept))
            macro.append(dict(factor=factor,method=method,seed_directions=len(rows),unique_seeds=5,mean_teacher_kl=kl,kl_ratio=kl/nk,correct=sum(r['correct'] for r in rows),directed_rows=sum(r['n'] for r in rows),teacher_label_agreement=statistics.mean(r['teacher_label_agreement'] for r in rows),mean_vector_error_ratio=statistics.mean(r['vector_error_ratio'] for r in rows),leave_one_seed_out_kl_ratio_min=min(leave),leave_one_seed_out_kl_ratio_max=max(leave)))
    for name,rows in [('correspondence_directions',summaries),('correspondence_blocks',blockrows),('functional_learning_curve',curves),('correspondence_macro',macro)]:
        (args.out/(name+'.json')).write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
        with (args.out/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    teachers=[]
    for seed in range(1,6):
        for factor in ['subject','distractor']:
            rows=[r for (s,f,_),r in source.items() if s==seed and f==factor and r['split']=='held_lexical_development'];teachers.append(dict(seed=seed,factor=factor,n=len(rows),correct=sum(r['donor_label_correct'] for r in rows),mean_noop_kl=statistics.mean(noop[(seed,factor,r['pair_id'])]['kl_to_reference'] for r in rows)))
    report=dict(run=str(args.run),sources=teachers,macro=macro,dependence='192prompts:16lexicalblocks; held subset8blocks×3syntax×4numberdirections. Five seeds produce20dependent ordered directions. No independent-direction CI; leave-one-seed-out range descriptive only.',ratio='KL(p_source_teacher||p_method) divided by KL(p_source_teacher||p_noop); sum losses before division; vector error uses common source-native16teacher.',status=json.loads((args.run/'status.json').read_text()),scope='All development; method chosen/changed using earlier pilot feedback; new data confirmation pending.')
    (args.out/'R2_RESULT_SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in macro:
        if row['factor']=='subject' and row['method'] in ['fcc_group16','fcc_native_units16','raw_native_units','full_code_native_units','full_code_ridge','single_atom','one_to_one','dense_native_select16','conditional_ot16','conditional_ot32']:print(json.dumps(row))


if __name__=='__main__':main()
