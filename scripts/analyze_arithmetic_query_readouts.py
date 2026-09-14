"""Compare member-field correspondence with same-information readout predictors."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/correspondence_reform_20260913'


def main():
    freeze = json.loads((ART/'R39_READOUT_EVAL_FREEZE_v2.json').read_text())
    path = ROOT/freeze['config_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == freeze['config_sha256']
    cfg = json.loads(path.read_text()); run = ROOT/'runs'/cfg['run_id']
    assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
    assert json.loads((run/'config.resolved.json').read_text()) == cfg
    with np.load(ART/'r39_member_confirmation.npz') as z:
        source, changed = z['source_aligned'], z['changed']
        ids = [tuple(x) for x in z['operand_pairs']]
        native = z['answers'][z['methods'].tolist().index('member')]
        baseline = z['baseline']
    primary = json.loads((ART/'r39_member_confirmation.json').read_text())
    panel = json.loads((run/'panel.json').read_text())
    assert panel == json.loads((ROOT/primary['run']/'panel.json').read_text())
    raw = run/'metrics.raw.jsonl'; digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert digest == json.loads((run/'metrics.summary.json').read_text())['metrics_raw_sha256']
    answers = {n: np.full(source.shape, -99999, dtype=np.int64) for n in ['raw', 'activation']}
    full = {n: np.full((source.shape[0],2,2,5,3), np.nan) for n in answers}
    records = [json.loads(s) for s in raw.read_text().splitlines()]
    base = {r['row_id']:r['answer'] for r in records if r['kind']=='base'}
    parsed = 0
    for r in records:
        if r['kind']!='source_patch' or r['seed']==0:
            continue
        p = panel['pairs'][r['row_id']]
        key = tuple(panel['rows'][i][k] for i in [p['recipient'],p['donor']] for k in ['a','b'])
        qi, oi, ti, si = ids.index(key), ['unit','tens'].index(r['operation']), p['template'], r['seed']-1
        assert baseline[qi,ti] == (-1 if base[p['recipient']] is None else base[p['recipient']])
        m = re.match(r'\s*(\d+)',r['generated_text']); value=int(m[1]) if m else -1
        assert value == (-1 if r['answer'] is None else r['answer'])
        recipient, donor = p['base_answer'], p['donor_answer']
        expected = recipient//10*10+donor%10 if oi==0 else donor//10*10+recipient%10
        valid = 10<=value<100
        target = valid and (value%10==donor%10 if oi==0 else value//10==donor//10)
        preserve = valid and (value//10==recipient//10 if oi==0 else value%10==recipient%10)
        flags = [value==expected,target,preserve]
        assert flags == [r[k] for k in ['exact_hybrid','target_digit_success','preserve_digit_success']]
        if r['method'].endswith('_full'):
            name=r['method'].split('_readout_')[0];ix=(qi,oi,ti,si)
            assert np.isnan(full[name][ix]).all();full[name][ix]=flags
        else:
            m=re.fullmatch(r'(raw|activation)_readout_part([01])_bank(\d+)',r['method']);assert m
            name,part,bank=m.groups();ix=(qi,int(bank),int(part),oi,ti,si)
            assert answers[name][ix]==-99999;answers[name][ix]=value
        parsed+=1
    assert all((a!=-99999).all() for a in answers.values())
    assert all(np.isfinite(a).all() for a in full.values())
    agreements={n:a==source for n,a in answers.items()};agreements['member']=native==source
    cells=[];counts={};axes=(2,3,4,5)
    den=np.stack([changed.sum(axes),(~changed).sum(axes)],-1)
    for name,a in agreements.items():
        cells.append(dict(method=name,exact_agreement=float(a.mean()),changed_agreement=float(a[changed].mean()),
                          unchanged_agreement=float(a[~changed].mean()),balanced_agreement=float(.5*(a[changed].mean()+a[~changed].mean()))))
        counts[name]=np.stack([(a&changed).sum(axes),(a&~changed).sum(axes)],-1)
    nq,nb=source.shape[:2];rng=np.random.default_rng(9391957)
    qw=rng.multinomial(nq,np.full(nq,1/nq),size=10000);bw=rng.multinomial(nb,np.full(nb,1/nb),size=10000)
    contrasts=[]
    for units,ww in [('question_clusters',np.ones_like(bw)),('question_and_partition',bw)]:
        dd=np.einsum('dq,db,qbc->dc',qw,ww,den)
        estimates={n:(np.einsum('dq,db,qbc->dc',qw,ww,c)/dd).mean(1) for n,c in counts.items()}
        for other in ['raw','activation']:
            point=next(c['balanced_agreement'] for c in cells if c['method']=='member')-next(c['balanced_agreement'] for c in cells if c['method']==other)
            contrasts.append(dict(comparator=other,units=units,difference_points=100*point,
                                  interval_points=(100*np.quantile(estimates['member']-estimates[other],[.025,.975])).tolist()))
    out=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),run=run.relative_to(ROOT).as_posix(),raw_sha256=digest,
             parsed_outputs=parsed,cells=cells,contrasts=contrasts,source_query_count=int(source.size),
             full_function=[dict(method=n,operation=op,**{m:float(a[:,o,...,k].mean()) for k,m in enumerate(['H','T','P'])}) for n,a in full.items() for o,op in enumerate(['unit','tens'])],
             scope=freeze['comparison'])
    (ART/'r39_query_readouts.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(ART/'r39_query_readouts.npz',methods=np.array(list(answers)),answers=np.stack(list(answers.values())),
                        source_aligned=source,changed=changed,full_function=np.stack(list(full.values())))
    print(json.dumps(out))


if __name__=='__main__':
    main()
