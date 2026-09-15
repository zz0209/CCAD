"""Keep changing-carry success and same-carry preservation jointly visible."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def main(run,output,seeds_filter=None):
    status=json.loads((run/'status.json').read_text());assert status['status']=='PASS'
    raw=run/'metrics.raw.jsonl';rows=[json.loads(s) for s in raw.read_text().splitlines()]
    baseline=[r for r in rows if r['kind']=='base']
    rows=[r for r in rows if r['kind']=='rule_intervention' and (seeds_filter is None or r['seed'] in seeds_filter)]
    panel=json.loads((run/'RULE_PANEL.json').read_text());expected={(p['recipient'],p['condition']):p for p in panel['pairs']}
    pair_metadata={p['component']:p for p in panel['canonical_pairs']}
    def bootstrap_weights(ids):
        labels=[pair_metadata[i].get('carry_group',pair_metadata[i].get('stratum','all')) for i in ids]
        rng=np.random.default_rng(9491502);weights=np.zeros((10000,len(ids)))
        for label in sorted(set(labels)):
            jj=[i for i,x in enumerate(labels) if x==label];n=len(jj)
            weights[:,jj]=rng.multinomial(n,np.full(n,1/n),10000)/len(ids)
        return weights
    methods=sorted({r['method'] for r in rows});seeds=sorted({r['seed'] for r in rows});arrays={};cells=[]
    conditions=['same_answer_opposite_carry','same_carry_different_answer']
    for method in methods:
        for condition in conditions:
            ids=sorted({p['component'] for p in panel['pairs'] if p['condition']==condition})
            rr=[r for r in rows if r['method']==method and r['operation']==condition]
            assert len(rr)==4*len(ids)*len(seeds)
            assert len({(r['seed'],r['row_id']) for r in rr})==len(rr)
            for r in rr:assert r['expected']==expected[r['row_id'],condition]['expected']
            values=np.array([[np.mean([float(r[k]) for r in rr if r['component']==cid])
                              for k in ['correct','unit_preserved','retained_original','edit_norm']] for cid in ids])
            weights=bootstrap_weights(ids)
            arrays[method+'|'+condition]=values
            cell=dict(method=method,condition=condition,canonical_pairs=len(ids),seeds=seeds)
            for i,name in enumerate(['success','unit_preserved','original_retained','edit_norm']):
                cell[name]=dict(mean=float(values[:,i].mean()),interval=np.quantile(weights@values[:,i],[.025,.975]).tolist())
            cells.append(cell)
    per_seed=[];per_stratum=[]
    for method in methods:
        for condition in conditions:
            for seed in seeds:
                rr=[r for r in rows if r['method']==method and r['operation']==condition and r['seed']==seed]
                per_seed.append(dict(method=method,condition=condition,seed=seed,requests=len(rr),
                                     success=float(np.mean([r['correct'] for r in rr])),
                                     unit_preserved=float(np.mean([r['unit_preserved'] for r in rr]))))
            labels=sorted({pair_metadata[r['component']].get('carry_group',condition) for r in rows if r['operation']==condition})
            for label in labels:
                rr=[r for r in rows if r['method']==method and r['operation']==condition
                    and pair_metadata[r['component']].get('carry_group',condition)==label]
                per_stratum.append(dict(method=method,condition=condition,stratum=label,seeds=seeds,
                    canonical_pairs=len({r['component'] for r in rr}),requests=len(rr),
                    success=float(np.mean([r['correct'] for r in rr])),
                    unit_preserved=float(np.mean([r['unit_preserved'] for r in rr]))))
    contrasts=[]
    for method,reference in [('field_fitted_weighted_64','conditional_carry_64'),
                             ('field_fitted_weighted_64','field_fitted_binary_64'),
                             ('field_fitted_weighted_64','field_ranked_binary_64'),
                             ('field_fitted_weighted_64','raw_carry_direction'),
                             ('raw_carry_direction','conditional_carry_64'),
                             ('field_fitted_binary_64','conditional_carry_64'),
                             ('field_fitted_binary_64','field_ranked_binary_64'),
                             ('field_fitted_binary_64','raw_carry_direction'),
                             ('function_ce_weighted_64','field_fitted_binary_64'),
                             ('function_ce_weighted_64','field_fitted_weighted_64'),
                             ('function_ce_weighted_64','raw_carry_direction'),
                             ('function_ce_binary_64','field_fitted_binary_64'),
                             ('function_ce_binary_64','raw_carry_direction'),
                             ('function_ce_binary_64','function_ce_weighted_64'),
                             ('transferred_fitted_binary_64','assignment_64'),
                             ('transferred_fitted_binary_64','field_fitted_binary_64'),
                             ('transferred_fitted_binary_64','transferred_fitted_weighted_64'),
                             ('transferred_fitted_binary_64','raw_carry_direction')]:
        if method not in methods or reference not in methods:continue
        for condition in conditions:
            difference=100*(arrays[method+'|'+condition][:,0]-arrays[reference+'|'+condition][:,0])
            ids=sorted({p['component'] for p in panel['pairs'] if p['condition']==condition})
            weights=bootstrap_weights(ids)
            contrasts.append(dict(method=method,reference=reference,condition=condition,
                                  difference_points=float(difference.mean()),interval_points=np.quantile(weights@difference,[.025,.975]).tolist()))
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),run=run.as_posix(),status=status,
                raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),cells=cells,per_seed=per_seed,per_stratum=per_stratum,contrasts=contrasts,
                baseline_accuracy=float(np.mean([r['correct'] for r in baseline])),baseline_prompts=len(baseline),
                selected_seeds=seeds,
                statistics='10,000 paired canonical-question-pair resamples within each condition and predeclared carry stratum, seed9491502; both orientations, prompt forms and dependent fixed SAE cohort remain together.',
                scope=panel.get('scope','Exposed development; changing-carry success and same-carry preservation are separate endpoints.'))
    output.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(output.with_suffix('.npz'),**arrays)
    print(json.dumps([dict(method=c['method'],condition=c['condition'],success=100*c['success']['mean']) for c in cells]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+')
    a=p.parse_args();main(a.run,a.output,a.seeds)
