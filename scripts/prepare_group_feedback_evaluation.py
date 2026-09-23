import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search-run', type=Path, required=True)
    parser.add_argument('--reference-selection', type=Path, required=True)
    parser.add_argument('--relation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert read(args.search_run/'status.json')['status'] == 'PASS'
    assert not args.output.exists()
    full = read(args.search_run/'selected_members.json')
    prefixes = read(args.search_run/'prefix_selected_members.json')
    reference = read(args.reference_selection)
    quota = dict(attn_2=2, resid_2=6, attn_3=6, resid_3=4, attn_4=2, resid_4=2)
    with np.load(args.relation, allow_pickle=False) as relation:
        candidates = {site:set(relation[site+'__candidates'].tolist()) for site in quota}
    total = sum(map(len,candidates.values()))
    selected = {f'{method}_n{budget}':prefixes[f'{method}_n{budget}']
                for method in ['adaptive', 'passive'] for budget in [32,64,128]}
    for method in ['adaptive','passive']:
        assert prefixes[f'{method}_n{total}'] == full[method]
        selected[method+'_nN'] = full[method]
    for method in ['finite_vector_n8', 'direct_calibration_n8']:
        selected[method] = reference[method]
    for groups in selected.values():
        assert set(groups)==set(quota)
        for site, members in groups.items():
            assert len(members)==len(set(members))==quota[site]
            assert set(members) <= candidates[site]
    args.output.mkdir(parents=True)
    (args.output/'selected_members.json').write_text(json.dumps(selected, indent=2)+'\n', encoding='utf-8')
    receipt = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), target_candidate_count=total,
        common_final_intervention_budget=total*16, groups=len(selected), quota=quota,
        prefix_budgets={str(n):n*16 for n in [32,64,128,total]},
        source_identity=[identity(args.search_run/name) for name in ['status.json','selected_members.json','prefix_selected_members.json']],
        reference_identity=identity(args.reference_selection), relation=identity(args.relation), generator=identity(Path(__file__)),
        selection='Every prespecified prefix and final support retained; no evaluation result used')
    (args.output/'PREPARATION.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
