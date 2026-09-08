"""Freeze a completed source procedure and natural-only baselines before new cities."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--procedure',type=Path,required=True);ap.add_argument('--atom-run',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--recovery',type=Path,help='Retained implementation-failure diagnosis; no method or endpoint tuning')
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError('Do not replace a frozen confirmation')
    procedure=json.loads(args.procedure.read_text());atom=args.atom_run.resolve()
    if json.loads((atom/'status.json').read_text())['status']!='PASS':raise ValueError('Atom calibration is not complete')
    if not json.loads((atom/'contract_validation.json').read_text())['ok']:raise ValueError('Atom calibration contract failed')
    panel=ROOT/procedure['confirmation_panel'];prepared=json.loads(panel.read_text());inventory=prepared['selection_inventory']
    records={};sources=[];sae_inputs=[]
    def add(path):
        p=Path(path);p=p if p.is_absolute() else ROOT/p
        rel=p.relative_to(ROOT).as_posix()
        records[rel]=dict(path=rel,sha256=digest(p),bytes=p.stat().st_size)
        return rel
    add(args.procedure);add(procedure['selection_comparison']);add(panel);add(procedure['old_panel']);add(procedure['raw_basis']);add(procedure['raw_singleton_basis'])
    for source in procedure['sources']:
        source=dict(source);run=ROOT/source['run'];status=json.loads((run/'status.json').read_text())
        if status['status']!='PASS' or not json.loads((run/'contract_validation.json').read_text())['ok']:
            raise ValueError('Source fit is not complete: '+source['run'])
        for p in ['config.resolved.json','code_hashes.json','family_source_summary.json']:add(run/p)
        source['atom_map']=add(atom/f"s{source['source_seed']}_t{source['target_seed']}_atom_pw_mcc.npz")
        add(source['basis_path']);add(source['replay_outputs'])
        sources.append(source)
        expected=Path(procedure['sae_root'])/f"seed_{source['source_seed']}"
        inputs=json.loads((run/'inputs.json').read_text())['inputs']
        for name in ['cfg.json','sae.safetensors']:
            path=expected/name;matches=[i for i in inputs if Path(i['path'])==path]
            if not matches:raise ValueError('Source material input identity missing')
            sha=digest(path)
            if sha!=matches[-1]['sha256']:raise ValueError('Source SAE changed after fitting')
            sae_inputs.append(dict(seed=source['source_seed'],path=str(path),bytes=path.stat().st_size,sha256=sha))
    for rel in procedure['evaluation_source_files']:add(rel)
    add(atom/'atom_selection.json');add(atom/'config.resolved.json');add(atom/'code_hashes.json')
    recovery=None
    if args.recovery:
        recovery=json.loads(args.recovery.read_text());add(args.recovery)
        for record in recovery['evidence_files']:add(record['path'])
    if len(sources)!=5 or {s['source_seed'] for s in sources}!=set(range(1,6)):
        raise ValueError('Exactly five independently trained source dictionaries required')
    if any(s['target_seed']!=s['source_seed']%5+1 for s in sources):raise ValueError('Expected fixed five-cycle direction order')
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        procedure_path=args.procedure.resolve().relative_to(ROOT).as_posix(),procedure_sha256=digest(args.procedure),
        old_panel=procedure['old_panel'],confirmation_panel=procedure['confirmation_panel'],raw_basis=procedure['raw_basis'],raw_singleton_basis=procedure['raw_singleton_basis'],
        confirmation_rows=inventory['confirmation_rows'],confirmation_city_pairs=inventory['maximum_disjoint_pairs'],
        sources=sources,files=list(records.values()),sae_inputs=sae_inputs,
        primary=procedure['primary'],secondary=procedure['secondary'],execution=procedure['execution'],recovery=recovery,
        scope='All method selection uses prior source calibration or natural calibration. New city intervention outcomes have not been read by this freezing script. Model-only capability screening has already occurred; this is confirmation within that population and the original RAVEL train templates, not official test.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(args.output.resolve()),sha256=digest(args.output),sources=len(sources),files=len(records),confirmation_rows=result['confirmation_rows'])))


if __name__=='__main__':main()
