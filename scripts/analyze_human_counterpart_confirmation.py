from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np
from analyze_request_trajectory import study,score


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    freeze=json.loads(a.freeze.read_text())
    for path,digest in freeze['identities'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError(path)
    results=[];raw=[];panels=[]
    for run in a.runs:
        result,heads,rows,families=study(run);results.append(result);raw.append(heads)
        panels.append(json.loads((run/'membership.json').read_text()))
    if any(v['human_rows']!=panels[0]['human_rows'] or v['human_query_order']!=panels[0]['human_query_order'] for v in panels):raise ValueError('Panel changed')
    if [json.loads((run/'config.resolved.json').read_text())['target_seed'] for run in a.runs]!=freeze['target_seeds']:raise ValueError('Targets changed')
    for heads in raw[1:]:
        if any(not np.array_equal(arrays['source'],raw[0][i][1]['source']) for i,(_,arrays) in enumerate(heads)):raise ValueError('Source changed')
    cells=[[i for i,r in enumerate(rows) if (r['profession'],r['gender'])==cell] for cell in sorted({(r['profession'],r['gender']) for r in rows})]
    winner='initial_counterpart_propagated_span_64';methods=list(results[0]['metrics'])
    samples={method:[] for method in methods if method!=winner};rng=np.random.default_rng(2026092115)
    for _ in range(2000):
        documents=np.concatenate([rng.choice(cell,len(cell),replace=True) for cell in cells]);scores={m:0. for m in methods}
        for heads in raw:
            for global_ids,arrays in heads:
                lookup={j:i for i,j in enumerate(global_ids)};selected=[lookup[j] for j in documents if j in lookup]
                for method in methods:scores[method]+=score(arrays[method],arrays['source'],arrays['none'],families['all'],selected)/(len(raw)*len(heads))
        for method in samples:samples[method].append(scores[method]-scores[winner])
    means={m:{f:float(np.mean([r['metrics'][m]['later'][f] for r in results])) for f in families} for m in methods}
    contrasts={m:dict(reduction=means[m]['all']-means[winner]['all'],ci95=np.quantile(v,[.025,.975]).tolist()) for m,v in samples.items()}
    result=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),studies=results,mean=means,comparisons=contrasts,
                documents=len(rows),statistics=freeze['statistics'],checks=dict(freeze_identity=True,same_panel=True,same_source=True))
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(mean=means,comparisons=contrasts),indent=2))


if __name__=='__main__':main()
