"""Separate natural reconstruction, source function and fixed-teacher transfer."""
import argparse, hashlib, json, statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_run(path):
    summary=json.loads((path/'metrics.summary.json').read_text())
    assert summary['status']=='PASS' and json.loads((path/'contract_validation.json').read_text())['ok']
    rows=[json.loads(line) for line in (path/'metrics.raw.jsonl').read_text().splitlines()]
    assert len(rows)==summary['rows']
    return dict(run=path.as_posix(),summary=summary,rows=rows,raw_sha256=hashlib.sha256((path/'metrics.raw.jsonl').read_bytes()).hexdigest())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training',required=True,type=Path);ap.add_argument('--function',required=True,type=Path);ap.add_argument('--old-transfer',required=True,type=Path);ap.add_argument('--new-transfer',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    training=read_run(args.training);function=read_run(args.function)
    assert len({r['quality']['ce']['clean'] for r in training['rows']})==1
    quality=[]
    for step in sorted({r['step'] for r in training['rows']}):
        rows=[r for r in training['rows'] if r['step']==step];assert len(rows)==5
        stats={}
        for k in ['fve','ce_recovered','actual_nonzero_l0','alive_features']:
            v=[r['quality'][k] for r in rows];stats[k]=dict(mean=statistics.fmean(v),min=min(v),max=max(v))
        stats['decoder_norm_max_error']=max(r['decoder_norm_max_error'] for r in rows)
        quality.append(dict(step=step,tokens=step*1024,**stats))
    groups=defaultdict(list)
    for r in function['rows']:groups[r['step'],r['factor']].append(r)
    fun=[]
    for (step,factor),rows in sorted(groups.items()):
        seeds=[]
        for seed in range(1,6):
            rr=[r for r in rows if r['seed']==seed]
            seeds.append(dict(seed=seed,n=len(rr),accuracy=statistics.fmean(r['correct'] for r in rr),kl_to_full_donor=statistics.fmean(r['kl_reference'] for r in rr)))
        fun.append(dict(step=step,factor=factor,n=len(rows),accuracy=statistics.fmean(r['accuracy'] for r in seeds),source_seed_means=seeds))
    old=json.loads(args.old_transfer.read_text());new=json.loads(args.new_transfer.read_text())
    key=lambda r:tuple(r[k] for k in ['factor','consumer','role','method'])
    lookup={key(r):r for r in old['rows']};changes=[]
    for row in new['rows']:
        before=lookup[key(row)]
        changes.append(dict(zip(['factor','consumer','role','method'],key(row)),old_kl=before['kl'],new_kl=row['kl'],kl_difference=row['kl']-before['kl'],relative_reduction=(1-row['kl']/before['kl']) if before['kl'] else None,directions_improved=sum(a['kl']>b['kl'] for a,b in zip(before['source_seed_means'],row['source_seed_means']))))
    invariant=[r for r in changes if r['method'] in ['legacy_contrast','raw_complete','raw_balanced']]
    assert max(abs(r['kl_difference']) for r in invariant)<1e-10
    # Check the actual fixed teacher distributions on every row, not just means.
    def refs(s):
        r=read_run(Path(s['run']))
        return {(x['source_seed'],x['factor'],x['consumer'],x['row_id']):x['label_logprobs'] for x in r['rows'] if x['method']=='source'}
    assert refs(old)==refs(new)
    phase=json.loads((args.training/'continuation_phase.json').read_text())
    result=dict(written_utc=datetime.now(timezone.utc).isoformat(),training=training,quality_by_step=quality,function_run=dict(run=function['run'],summary=function['summary'],raw_sha256=function['raw_sha256']),source_function=fun,transfer_changes=changes,fixed_teacher_label_distributions_exact=True,fixed_legacy_and_raw_kl_max_error=max(abs(r['kl_difference']) for r in invariant),continuation=phase,scope=f"Training continues for {phase['additional_steps']} additional updates with fresh natural tokens and the recorded restarted LR. The previous stream prefix is not retrained. Natural quality uses the original256 validation sequences; source function reselects groups on old384 temporal prompts. Transfer alone holds the4M source teachers and all inputs fixed while replacing target codes. All evaluation remains exposed development, five dependent cyclic directions.")
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(quality=quality,function=fun,ols_changes=[r for r in changes if r['method']=='ols_complete']),indent=2))


if __name__=='__main__':main()
