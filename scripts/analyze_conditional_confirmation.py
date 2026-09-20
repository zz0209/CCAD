from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
from analyze_profile_confirmation import human_arrays, scalar_arrays, query_order, infer


ART=Path('artifacts/final_science_20260920_round02')
BULK=Path('D:/CCAD_Storage/runs/final_science_20260920_round02')
PROFILE='initial_refined_conditional_source_metric'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    freeze=json.loads((ART/'CONDITIONAL_CONFIRMATION_FREEZE.json').read_text())
    for item in freeze['inputs']:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    seeds=list(range(1,6))
    runs=[BULK/f'CONDITIONAL_CONFIRM_T{s}_20260920' for s in seeds]
    for run in runs:
        assert json.loads((run/'status.json').read_text())['status']=='PASS',run
        for file in ['scripts/train_intervention_changes.py','src/ccad/intervention_transport.py']:
            assert hashlib.sha256((run/'source_snapshot'/file).read_bytes()).hexdigest()==hashlib.sha256(Path(file).read_bytes()).hexdigest()
    memberships=[json.loads((r/'membership.json').read_text()) for r in runs]
    assert all(m==memberships[0] for m in memberships)
    names=query_order(runs[0],'human')
    assert all(query_order(r,'human')==names for r in runs)
    mapping=json.loads((ART/'CONDITIONAL_CONFIRMATION_REQUESTS.json').read_text())['human_families']
    assert set(mapping)==set(names)
    families={f:[i for i,n in enumerate(names) if mapping[n]==f] for f in sorted(set(mapping.values()))}
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),freeze=freeze,queries=names,
                inference='Paired seed/document/request intervals, conditional on the fixed source program and four later heads')
    for label,selected,ids in [('all_targets',runs,seeds),('trained_comparison',runs[1:],seeds[1:])]:
        result[label]={}
        nums,den,dw,rng,extra=human_arrays(selected,memberships[0])
        result[label]['later_heads']=infer(nums,den,dw,families,ids,rng,profile=PROFILE)
        result[label]['later_heads']['provenance']=extra
        nums,den,dw,rng,_=scalar_arrays(selected,memberships[0],'human')
        result[label]['original_head']=infer(nums,den,dw,families,ids,rng,profile=PROFILE)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for label in result:
        if label not in ['all_targets','trained_comparison']: continue
        print(json.dumps({label:{h:{m:v['participation'] for m,v in d['summary'].items()} for h,d in result[label].items()}},indent=2))


if __name__=='__main__': main()
