"""Development test of reusing calibration responses for later decisions.

All selectors receive the same seven calibration requests and documents.
They differ in whether they retain the old scalar or the full response.
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import sys


def main():
    root=Path('artifacts/science_upgrade_20260919')
    anchor_mode='--interior-anchor' in sys.argv
    matched='--matched-requests' in sys.argv
    output=root/('INTERIOR_ANCHOR_SELECTION_MATCHED.json' if anchor_mode and matched else
                 'INTERIOR_ANCHOR_SELECTION.json' if anchor_mode else
                 'VERTEX_SELECTION_MATCHED.json' if matched else 'REQUEST_READOUT_SELECTION.json')
    if output.exists():
        raise FileExistsError(output)
    calibration=[Path('runs/SCIENCE01_response_space_dev_v1_20260919'),
                 Path('runs/SCIENCE01_response_continuous_dev_v1_20260919')]
    evaluation=[Path('runs/SCIENCE01_response_space_tasks_v1_20260919'),
                Path('D:/CCAD_Storage/runs/science_upgrade_20260919/SCIENCE01_response_continuous_tasks_v2_20260919')]
    cfg=json.loads((calibration[0]/'config.resolved.json').read_text())
    groups=['pronouns','names','associated_words']
    methods=['initial','head_parts','pooled_parts','white_parts','pooled_whole',
             'head_continuous','pooled_continuous']
    tasks=['composer_surgeon_orientation0','composer_surgeon_orientation1',
           'model_software_engineer_orientation0','model_software_engineer_orientation1']
    provenance=[]

    def load(roots,name):
        path=next(p/name for p in roots if (p/name).exists())
        provenance.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return np.load(path).astype(np.float64)

    cal_rows=json.loads((calibration[0]/'evaluation_membership.json').read_text())['rows']
    for p in calibration[1:]:
        assert json.loads((p/'evaluation_membership.json').read_text())['rows']==cal_rows
    eval_rows=json.loads((evaluation[0]/'evaluation_membership.json').read_text())['rows']
    for p in evaluation[1:]:
        assert json.loads((p/'evaluation_membership.json').read_text())['rows']==eval_rows
    assert not {r['document_sha256'] for r in cal_rows}&{r['document_sha256'] for r in eval_rows}
    w0=np.load(Path(cfg['frozen_source_run'])/'probe.npz')['weight'].ravel().astype(np.float64)
    heads={t:np.load(Path('runs/IR04_shift_consumer_seed2_v1_20260916')/f'none__full__{t}__probe42.npz')['weight'].ravel().astype(np.float64) for t in tasks}
    clean=load(calibration,'none__full__pooled.npy')
    clean_eval=load(evaluation,'none__full__pooled.npy')
    vertices=[]
    for mask in range(8):
        bits=np.array([(mask>>i)&1 for i in range(3)])
        query='full' if mask==7 else '+'.join(g for i,g in enumerate(groups) if bits[i])
        vertices.append((bits,query))
    sc={q:(load(calibration,f'source__{q}__pooled.npy')-clean if q else np.zeros_like(clean)) for _,q in vertices}
    mc={m:{q:(load(calibration,f'{m}__{q}__pooled.npy')-clean if q else np.zeros_like(clean)) for _,q in vertices} for m in methods}
    rows=[]
    anchor_source=load(calibration,'source__mix_balanced__pooled.npy') if anchor_mode else None
    # Later requests are fixed fractional requests, absent from these selectors'
    # seven-vertex calibration. The programs' training families are reported separately.
    for query,qs in cfg['dose_queries'].items():
        if (anchor_mode or matched) and query=='mix_balanced': continue
        q=np.array([qs[g] for g in groups])
        coeff={v:float(np.prod(np.where(bits,q,1-q))) for bits,v in vertices}
        source_cal=sum(coeff[v]*sc[v] for _,v in vertices)
        if anchor_mode: source_cal=anchor_source-clean
        source_test=load(evaluation,f'source__{query}__pooled.npy')
        for task,w in heads.items():
            actual=[];old=[];full=[];pooled=[]
            for method in methods:
                err_cal=sum(coeff[v]*(mc[method][v]-sc[v]) for _,v in vertices)
                if anchor_mode:
                    err_cal=load(calibration,f'{method}__mix_balanced__pooled.npy')-anchor_source
                target=load(evaluation,f'{method}__{query}__pooled.npy')
                effect_test=(source_test-clean_eval)@w
                task_professions=(5,25) if task.startswith('composer') else (12,24)
                ix=np.array([r['profession'] in task_professions for r in eval_rows])
                actual.append(float(np.linalg.norm(((target-source_test)@w)[ix])/np.linalg.norm(effect_test[ix])))
                old.append(float(np.linalg.norm(err_cal@w0)/np.linalg.norm(source_cal@w0)))
                full.append(float(np.linalg.norm(err_cal@w)/np.linalg.norm(source_cal@w)))
                pooled.append(float(np.linalg.norm(err_cal)/np.linalg.norm(source_cal)))
            for selector,scores in [('old_scalar',old),('full_response',full),('pooled_norm',pooled)]:
                choice=int(np.argmin(scores));oracle=int(np.argmin(actual))
                rows.append(dict(request=query,head=task,selector=selector,selected=methods[choice],
                                 oracle=methods[oracle],actual_nrmse=actual[choice],oracle_nrmse=actual[oracle],
                                 regret=actual[choice]-actual[oracle],predicted=scores,actual=actual))
    summary={s:dict(mean_nrmse=float(np.mean([r['actual_nrmse'] for r in rows if r['selector']==s])),
                    mean_regret=float(np.mean([r['regret'] for r in rows if r['selector']==s])),
                    choices={m:sum(r['selected']==m for r in rows if r['selector']==s) for m in methods})
             for s in ['old_scalar','full_response','pooled_norm']}
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),evidence='Adaptive exposed development only; fixed correlated request/head cells, one target seed',
                calibration_documents=len(cal_rows),evaluation_documents=len(eval_rows),
                target_queries_per_method=(1 if anchor_mode else 7)*len(cal_rows),methods=methods,
                evaluation_scope='Each task head restricted to its own profession pair',
                selection_rule='Use the balanced interior request to select one adapter; evaluate only the other three fractional requests' if anchor_mode else 'Interpolate the seven calibration vertices',
                qualification='Interior-anchor selection was proposed after the vertex-selection failure and is exploratory.' if anchor_mode else 'Multiaffine interpolation of complete executed responses is an approximation. Exact identity only applies when the response is multiaffine.',
                summary=summary,rows=rows,inputs=provenance)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
