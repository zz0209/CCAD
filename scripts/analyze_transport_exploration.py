"""Compare standalone transport development runs on paired original outputs."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BULK=Path('D:/CCAD_Storage/runs/transport_exploration_20260919')
OUT=ROOT/'artifacts/transport_exploration_20260919'


def grammar():
    run=BULK/'TRANSPORT01_infinitive_development_v1_20260919'
    idx=json.loads((run/'INDEX.json').read_text());v=dict(np.load(run/'responses.npz'))
    queries=idx['queries'];den=((v['source'].astype(float)-v['none'])**2).mean(1)
    extra=BULK/'TRANSPORT01_initial_open_grammar_v1_20260919'
    if (extra/'status.json').exists():
        ex=dict(np.load(extra/'responses.npz'))
        assert np.array_equal(ex['source'],v['source']) and np.array_equal(ex['none'],v['none'])
        v['transport_open_initial']=ex['transport_open_initial']
        v['transport_refined_open']=ex['transport_refined_open']
    families={'all':list(range(len(queries))),'endpoints':[0,1,2],
              'held_requests':[i for i,q in enumerate(queries) if q.startswith(('interior','boundary'))]}
    summary={}
    for m,a in v.items():
        if m in ('source','none'):continue
        e=np.sqrt(((a.astype(float)-v['source'])**2).mean(1)/den)
        summary[m]={f:float(e[ix].mean()) for f,ix in families.items()}
    return dict(summary=summary,quality=json.loads((run/'quality.json').read_text()),queries=queries,
                evidence='One target seed, exposed development texts/requests; source fixed.',run=str(run))


def human():
    run=BULK/'TRANSPORT01_human_program_development_v1_20260919'
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    old=Path('D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE03_shift_execution_reform_v1_20260919')
    members=json.loads((run/'evaluation_membership.json').read_text())
    assert members==json.loads((old/'evaluation_membership.json').read_text())
    assert json.loads((run/'program_context_membership.json').read_text())==json.loads((old/'program_context_membership.json').read_text())
    config=json.loads((ROOT/'configs/transport01_human_program_development_v1.json').read_text());queries=config['queries']
    rows=members['rows'];clean=np.load(run/'none__full__pooled.npy').astype(float)
    source=np.stack([np.load(run/f'source__{q}__pooled.npy').astype(float) for q in queries])
    assert np.array_equal(clean,np.load(old/'none__full__pooled.npy'))
    for i,q in enumerate(queries):assert np.array_equal(source[i],np.load(old/f'source__{q}__pooled.npy'))
    tasks=['composer_surgeon_orientation0','composer_surgeon_orientation1','model_software_engineer_orientation0','model_software_engineer_orientation1']
    heads=[np.load(ROOT/'runs/IR04_shift_consumer_seed2_v1_20260916'/f'none__full__{t}__probe42.npz') for t in tasks]
    weights=np.stack([h['weight'].ravel() for h in heads],1)
    bias=np.array([h['bias'].item() for h in heads])
    masks=[np.array([r['profession'] in p for r in rows]) for p in [(5,25),(5,25),(12,24),(12,24)]]
    families={'all':list(range(len(queries))),'endpoints':list(range(7)),
              'held_requests':[i for i,q in enumerate(queries) if q.startswith(('interior','boundary'))]}
    sources={m:run for m in ['initial','input_initial','transport_open_initial','raw_reconstruction','transport_active_mixed','transport_open_mixed']}
    sources.update({m:old for m in ['tangent_gain','tangent_mixed']})
    summary={};effect=[];perquery={}
    for m,r in sources.items():
        t=np.stack([np.load(r/f'{m}__{q}__pooled.npy').astype(float) for q in queries]);errors=[];agreement=[];accuracy=[]
        for h,ix in enumerate(masks):
            den=(((source[:,ix]-clean[ix])@weights[:,h])**2).mean(1)
            errors.append(np.sqrt((((t[:,ix]-source[:,ix])@weights[:,h])**2).mean(1)/np.maximum(den,1e-12)))
            tl=t[:,ix]@weights[:,h]+bias[h];sl=source[:,ix]@weights[:,h]+bias[h]
            labels=np.array([r['profession']==(25 if h<2 else 24) for r,k in zip(rows,ix) if k])
            agreement.append(((tl>0)==(sl>0)).mean(1));accuracy.append(((tl>0)==labels).mean(1))
            if m=='initial':effect.append(np.sqrt(den).tolist())
        e=np.array(errors);a=np.array(agreement);acc=np.array(accuracy)
        summary[m]={f:dict(nrmse=float(e[:,ix].mean()),source_agreement=float(a[:,ix].mean()),task_accuracy=float(acc[:,ix].mean())) for f,ix in families.items()}
        perquery[m]=e.tolist()
    return dict(summary=summary,per_head_query=perquery,source_effect_rms=effect,queries=queries,heads=tasks,
                contexts=len(rows),matched_evaluation_and_training=True,source_arrays_bit_equal=True,
                evidence='Fixed source, one target seed, exposed development texts and requests. No population significance claim.',run=str(run),prior_run=str(old))


def heldout():
    runs={}
    for label,rid in [('grammar_fit','TRANSPORT01_heldout_human_v2_20260919'),('natural_fit','TRANSPORT01_coverage_human_eval_v1_20260919'),('refined','TRANSPORT01_refined_human_v1_20260919')]:
        run=BULK/rid;metric=[json.loads(l) for l in (run/'metrics.raw.jsonl').read_text().splitlines()]
        runs[label]={m:{k:float(np.mean([r[k] for r in metric if r['method']==m])) for k in ['nrmse','hidden_nrmse','source_rms','source_flip_rate']} for m in sorted({r['method'] for r in metric})}
    one=np.load(BULK/'TRANSPORT01_heldout_human_v1_20260919/responses.npz')
    two=np.load(BULK/'TRANSPORT01_heldout_human_v2_20260919/responses.npz')
    assert one.files==two.files and all(np.array_equal(one[k],two[k]) for k in one.files)
    runs['v1_v2_response_arrays_bit_equal']=True
    return runs


if __name__=='__main__':
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),grammar=grammar(),human=human(),heldout=heldout(),
                analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=OUT/'DEVELOPMENT_ANALYSIS.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['summary'] for k,v in result.items() if isinstance(v,dict) and 'summary' in v},indent=2))
