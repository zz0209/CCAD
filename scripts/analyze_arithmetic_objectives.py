"""Compare correspondence objectives with shared question/SAE dependence."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/correspondence_reform_20260913'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--panel',choices=['original','range'],default='original')
    args=parser.parse_args()
    groups, identities, inputs = {}, None, []
    specs = [
        ('REFORM_R37_qwen_role_transfer_v1_20260914',
         {'counterfactual_weighted':'source','transfer_clean':'full_field','transfer_assignment':'assignment'}),
        ('REFORM_R38_qwen_task_metric_five_v1_20260914', {'transfer_clean':'conditional'}),
        ('REFORM_R38_qwen_requested_only_v1_20260914', {'transfer_clean':'requested_only'}),
        ('REFORM_R38_qwen_random_same_rank_v1_20260914', {'transfer_clean':'random_rank'}),
        ('REFORM_R38_qwen_member_fields_five_v1_20260914', {'transfer_clean':'member_fields'}),
    ]
    if args.panel=='range':
        specs=[
            ('REFORM_R37_qwen_role_transfer_confirmation_v1_20260914',
             {'role_source':'source','role_field':'full_field','role_assignment':'assignment'}),
            ('REFORM_R38_qwen_objective_range_replay_v1_20260914',
             {'conditional':'conditional','requested_only':'requested_only','random_rank':'random_rank'})]
    nparsed = 0
    for run, methods in specs:
        p = ROOT/'runs'/run
        if not (p/'metrics.summary.json').exists():
            continue
        assert json.loads((p/'status.json').read_text())['status'] == 'PASS'
        raw = p/'metrics.raw.jsonl'
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        assert digest == json.loads((p/'metrics.summary.json').read_text())['metrics_raw_sha256']
        inputs.append(dict(run=str(p.relative_to(ROOT)), raw_sha256=digest))
        panel = json.loads((p/'panel.json').read_text())
        cfg = json.loads((p/'config.resolved.json').read_text())
        if args.panel=='original':
            assert cfg['relation_transfer']['seed_pairs'] == [[1,2],[2,3],[3,4],[4,5],[5,1]]
        else:
            assert cfg['frozen_adaptation']['source_seed_for_target']=={'1':5,'2':1,'3':2,'4':3,'5':4}
        ids = [[panel['rows'][i][k] for i in [q['recipient'],q['donor']] for k in ['a','b']]
               for q in panel['pairs'][:64]]
        if identities is None: identities = ids
        assert ids == identities
        for line in raw.read_text().splitlines():
            r = json.loads(line)
            if r['kind'] != 'source_patch' or not r['seed']: continue
            method = re.sub(r'^transfer_s\d+_', 'transfer_', r['method'])
            if method.startswith('adapt_'):
                method=method[6:].rsplit('_u',1)[0]
            if method not in methods: continue
            name = methods[method]
            if name not in groups: groups[name] = np.full((64,2,2,5,3), np.nan)
            q = panel['pairs'][r['row_id']]
            rec, donor = [panel['rows'][q[k]] for k in ['recipient','donor']]
            cluster = [v[k] for v in [rec,donor] for k in ['a','b']]
            ix = (identities.index(cluster), ['unit','tens'].index(r['operation']),q['template'],r['seed']-1)
            assert np.isnan(groups[name][ix]).all()
            # Independent decimal parsing and label derivation from operands.
            match = re.search(r'\d+', r['generated_text'])
            answer = int(match.group()) if match else None
            requested = r['operation']
            expected = rec['total']//10*10+donor['total']%10 if requested=='unit' else donor['total']//10*10+rec['total']%10
            h = answer == expected
            t = answer is not None and (answer%10==donor['total']%10 if requested=='unit' else answer//10==donor['total']//10)
            preserve = answer is not None and (answer//10==rec['total']//10 if requested=='unit' else answer%10==rec['total']%10)
            observed = [r[k] for k in ['exact_hybrid','target_digit_success','preserve_digit_success']]
            assert [h,t,preserve] == observed
            groups[name][ix] = observed
            nparsed += 1
    assert all(np.isfinite(a).all() for a in groups.values())
    cells=[]
    for name,a in groups.items():
        for oi,op in enumerate(['unit','tens']):
            cells.append(dict(method=name, operation=op, n=640,
                              **{m:float(a[:,oi,...,k].mean()) for k,m in enumerate(['H','T','P'])},
                              per_seed_H=a[:,oi,:,:,0].mean((0,1)).tolist()))
    draws = np.random.default_rng(9380914).integers(64,size=(10000,64))
    contrasts=[]
    comparisons=[('conditional',other) for other in ['full_field','assignment','requested_only','random_rank']]
    comparisons += [('member_fields',other) for other in ['full_field','assignment','conditional']]
    for reference,other in comparisons:
        if other not in groups or reference not in groups: continue
        for op,sel in [('both',slice(None)),('unit',0),('tens',1)]:
            diff = groups[reference][:,sel]-groups[other][:,sel]
            delta = diff.mean(tuple(range(1,diff.ndim-1)))
            ci = np.quantile(delta[draws].mean(1),[.025,.975],axis=0)*100
            contrasts.append(dict(reference=reference,comparator=other,operation=op,
                metrics={m:dict(difference_points=float(delta[:,k].mean()*100),interval_points=ci[:,k].tolist())
                         for k,m in enumerate(['H','T','P'])}))
    out=dict(written_utc=datetime.now(timezone.utc).isoformat(),
             scope=f'Exploratory exposed {args.panel} development; intervals resample64questionclusters with two prompts/two requests/five fixed dependentSAEs.',
             cells=cells,contrasts=contrasts,inputs=inputs,independently_parsed_texts=nparsed)
    stem='r38_objectives'+('_range' if args.panel=='range' else '')
    (ART/f'{stem}.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(ART/f'{stem}.npz',outcomes=np.stack(list(groups.values())),
                        methods=np.array(list(groups)),operand_pairs=np.array(identities))
    print(json.dumps(dict(means={n:float(a[...,0].mean()) for n,a in groups.items()},
                         contrasts=[c for c in contrasts if c['operation']=='both'])))


if __name__ == '__main__': main()
