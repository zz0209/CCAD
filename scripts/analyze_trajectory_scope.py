from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import numpy as np
from analyze_request_trajectory import study


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result,_,_,_=study(args.run)
    current=json.loads((args.run/'membership.json').read_text())
    reference=json.loads((args.reference/'membership.json').read_text())
    row_index=[reference['human_rows'].index(row) for row in current['human_rows']]
    shared_queries=[query for query in current['human_query_order'] if query in reference['human_query_order']]
    if not shared_queries:
        raise ValueError('No shared requests to replay')
    current_index=[current['human_query_order'].index(query) for query in shared_queries]
    query_index=[reference['human_query_order'].index(query) for query in shared_queries]
    checks={}
    for filename,prefix in [('responses.npz','human__'),('pooled.npz','')]:
        with np.load(args.run/filename) as actual, np.load(args.reference/filename) as expected:
            for name in ['none','source','initial_trajectory_feedback']:
                key=prefix+name
                desired=expected[key][np.ix_(query_index,row_index)]
                gap=float(np.max(np.abs(actual[key][current_index].astype(float)-desired.astype(float))))
                checks[filename+'__'+name]=gap
                if gap!=0:
                    raise ValueError(f'Replay changed {filename} {name} by {gap}')
    diagnostics=json.loads((args.run/'execution_diagnostics.json').read_text())
    scoped={}
    for row in diagnostics:
        if row['dataset']!='human' or not row['sites']:
            continue
        total=scoped.setdefault(row['method'],dict(states=0,changed=0,excluded_states=0,excluded_changes=0))
        for values in row['sites'].values():
            for key in total:
                total[key]+=values.get(key,0)
    for values in scoped.values():
        if values['excluded_changes']!=0:
            raise ValueError('An excluded source location was changed')
        values['changed_per_site_token']=values['changed']/values['states']
    result.update(written_at_utc=datetime.now(timezone.utc).isoformat(),replay_max_absolute_error=checks,replay_queries=shared_queries,
                  execution=scoped,scope='Original development biographies and target 2. Same source request, dictionaries, upper member allowance and solver. Source locations are selected without target outcomes.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(metrics=result['metrics'],execution=scoped,replay=checks),indent=2))


if __name__=='__main__':
    main()
