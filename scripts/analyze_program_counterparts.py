from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bank = json.loads((args.selection/'counterparts.json').read_text())
    selection = json.loads((args.selection/'source_design.json').read_text())
    evaluation = json.loads((args.evaluation/'source_design.json').read_text())
    for run in [args.selection, args.evaluation]:
        if json.loads((run/'status.json').read_text())['status'] != 'PASS':
            raise ValueError(f'Incomplete run {run}')
    if selection['selection'] != evaluation['selection']:
        raise ValueError('Source definition changed')
    select_text = {r['clean_prefix'] for r in selection['rows']}
    eval_text = {r['clean_prefix'] for r in evaluation['rows']}
    arrays = dict(np.load(args.evaluation/'responses.npz'))
    source_energy = {q:float(np.mean((arrays[q]-arrays['none'])**2)) for q in ['singular','plural','full']}
    metrics = json.loads((args.evaluation/'target_analysis.json').read_text())
    summary = {}
    for method, values in metrics['metrics'].items():
        entry = dict(mean_nrmse=float(np.mean(list(values.values()))), requests=values,
                     singleton_mean=float(np.mean([values[q] for q in ['singular','plural']])),
                     unused_union=values['full'])
        if method in metrics['execution']:
            cost = metrics['execution'][method]
            entry['mean_changed_members'] = cost['changed']/cost['states']
        summary[method] = entry
    supports = {}
    for method, parts in bank['supports'].items():
        per_site = {}
        for site in parts['singular']:
            first, second = set(parts['singular'][site]), set(parts['plural'][site])
            union = first | second
            per_site[site] = dict(singular=len(first), plural=len(second), union=len(union),
                overlap=len(first & second), jaccard=len(first & second)/len(union) if union else None)
        supports[method] = dict(sites=per_site, total_union=sum(r['union'] for r in per_site.values()))
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
                  selection_run=str(args.selection), evaluation_run=str(args.evaluation),
                  selection_documents=len(selection['rows']), evaluation_documents=len(evaluation['rows']),
                  overlapping_prefixes=len(select_text & eval_text), selected_requests=bank['selected_requests'],
                  source_energy=source_energy, methods=summary, support=supports,
                  statistics='Development comparison on previously exposed prefixes. No population interval or confirmation claim.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(methods=summary, overlapping_prefixes=result['overlapping_prefixes']), indent=2))


if __name__ == '__main__':
    main()
