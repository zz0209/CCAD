"""Compare all completed source candidates; selection uses old calibration only."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from statistics import fmean

ROOT=Path(__file__).resolve().parents[1]
CONTROLS=['100','010','001','110','101','011','111']
LABELS={'decoded_independent':('Decoded / independent','Decoded / ind.'),
    'decoded_common':('Decoded / common blocks','Decoded / shared'),
    'raw_independent':('Raw / independent','Raw / ind.'),'raw_common':('Raw / common blocks','Raw / shared'),
    'decoded_singleton_reference':('Decoded / prior singleton fit','Decoded / prior'),
    'raw_singleton_reference':('Raw / prior singleton fit','Raw / prior')}


def measured_cells(rows,method):
    cells=[]
    for op in CONTROLS:
        rr=[r for r in rows if r['kind']=='semantic' and r['method']==method and r['operation']==op]
        if not rr:raise ValueError('Missing measured source control: '+method+'/'+op)
        cause=[r['first_token_correct'] for r in rr if r['endpoint']=='Cause'];iso=[r['first_token_correct'] for r in rr if r['endpoint']=='Iso']
        ca=fmean(cause) if cause else None;isolation=fmean(iso) if iso else None
        cells.append(dict(operation=op,cause=ca,iso=isolation,score=fmean(v for v in [ca,isolation] if v is not None),rows=len(rr)))
    return cells


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    run=args.run.resolve();inputs=[]
    def read(p):
        p=Path(p);p=p if p.is_absolute() else ROOT/p;inputs.append(p.relative_to(ROOT).as_posix());return json.loads(p.read_text())
    def raw(p):
        summary=read(p/'metrics.summary.json')
        if summary['status']!='PASS' or not read(p/'contract_validation.json')['ok']:raise ValueError('Source comparison requires completed runs')
        payload=(p/'metrics.raw.jsonl').read_bytes();inputs.append((p/'metrics.raw.jsonl').relative_to(ROOT).as_posix())
        if hashlib.sha256(payload).hexdigest()!=summary['metrics_raw_sha256']:raise ValueError('Source raw identity differs')
        return [json.loads(line) for line in payload.splitlines()]
    cfg=read(run/'config.resolved.json');rows=raw(run);sources=read(run/'family_source_summary.json')['methods']
    prior=ROOT/'runs/FINAL5_R17_l7_projected_transfer_v1_20260908';oldrows=raw(prior)
    methods=[]
    for r in sources:
        name=r['method'];cal=r['selected'];held=measured_cells(oldrows,{'decoded_singleton_reference':'frozen_source','raw_singleton_reference':'raw_supervised_das'}[name]) if 'source_reference' in r else measured_cells(rows,name)
        if 'source_reference' in r:
            fit=read(ROOT/r['source_reference']['run']/(r['source_reference']['kind']+'_fit.json'))
            updates=fit['updates'];step=fit['selected_step'];dof=None
        else:updates=r['updates'];step=r['selected_step'];dof=r['stiefel_degrees_of_freedom']
        methods.append(dict(method=name,label=LABELS[name][0],short_label=LABELS[name][1],
            source_seed=cfg['source_seed'],updates=updates,selected_step=step,parameter_array_count=r.get('parameter_count',3*cfg['das_rank']*2048),stiefel_degrees_of_freedom=dof,
            calibration_family=cal['family_score'],calibration_single=cal['single_score'],calibration_loss=cal['loss'],
            calibration_controls=[dict(operation=''.join(map(str,c['control'])),**c) for c in cal['cells']],
            held_family=fmean(c['score'] for c in held),held_single=fmean(c['score'] for c in held[:3]),held_controls=held,
            basis_path=(run/r['basis_path']).relative_to(ROOT).as_posix(),
            within_orthogonality_error=r.get('within_orthogonality_error'),full_orthogonality_error=r.get('full_orthogonality_error'),
            common_blocks=r['common_blocks']))
    def choose(prefix):
        # Python min keeps the configured candidate order after a complete tie.
        return min([r for r in methods if r['method'].startswith(prefix)],key=lambda r:(-r['calibration_family'],-r['calibration_single'],r['calibration_loss']))['method']
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),methods=methods,
        selected_decoded=choose('decoded'),selected_raw=choose('raw'),
        selection_rule='Maximum seven-request source calibration score, then maximum singleton score, then minimum balanced CE; exact ties keep the original candidate order. Old held outcomes are reported, never used in this choice.',
        input_summary_paths=list(dict.fromkeys(inputs)),
        scope='All outcomes here are exposed source1 development. Four new fits share1792updates/rank512; prior1536update singleton-trained choices remain additional candidates, not equal-update attribution controls. Shared blocks have fewer Stiefel degrees of freedom. Raw and decoded selections are separate.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output.resolve()),selected_decoded=result['selected_decoded'],selected_raw=result['selected_raw'],methods=[{k:r[k] for k in ['method','calibration_family','held_single','held_family']} for r in methods])))


if __name__=='__main__':main()
