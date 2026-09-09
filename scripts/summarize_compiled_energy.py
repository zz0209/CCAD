"""Describe all retained development norm/orientation cells without selection."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json


def summarize(run):
    assert json.loads((run/'status.json').read_text())['status']=='PASS'
    config=json.loads((run/'config.resolved.json').read_text())
    result=json.loads((run/'query_results.json').read_text())
    assert [r['query'] for r in result['queries']]==config['queries']
    comparisons=[]; cells=[]
    for r in result['queries']:
        s=r['summary']; g=s['geometric_original']; gm=s['geometric_learned_norm']
        fd=s['functional_original_norm']; f=s['functional_learned_norm']
        contrasts={}
        for endpoint in ['iia','source_kl','margin','donor_ce']:
            contrasts[endpoint]=dict(
                total=f[endpoint]-g[endpoint],
                norm_at_geometric_direction=gm[endpoint]-g[endpoint],
                norm_at_functional_direction=f[endpoint]-fd[endpoint],
                direction_at_original_norm=fd[endpoint]-g[endpoint],
                direction_at_learned_norm=f[endpoint]-gm[endpoint],
                interaction=f[endpoint]-fd[endpoint]-gm[endpoint]+g[endpoint])
        comparisons.append(dict(query=r['query'],task=r['task'],objective=r['objective'],
                                held_rows=r['held_rows'],contrasts=contrasts))
        cells.extend(dict(query=r['query'],task=r['task'],objective=r['objective'],
                          method=k,held_rows=r['held_rows'],**v) for k,v in s.items())
    def mean(values):return sum(values)/len(values)
    macro={endpoint:{key:mean([r['contrasts'][endpoint][key] for r in comparisons])
                     for key in comparisons[0]['contrasts'][endpoint]}
           for endpoint in comparisons[0]['contrasts']}
    method_means={method:{endpoint:mean([r[endpoint] for r in cells if r['method']==method])
                         for endpoint in ['iia','source_kl','margin','donor_ce','target_delta_norm']}
                  for method in config['variants']}
    return dict(written_at_utc=datetime.now(timezone.utc).isoformat(),queries=len(comparisons),
        comparisons=comparisons,method_means=method_means,macro_contrasts=macro,cells=cells,
        inputs=[dict(path=str(run/name),sha256=hashlib.sha256((run/name).read_bytes()).hexdigest())
                for name in ['config.resolved.json','query_results.json','metrics.raw.jsonl','metrics.summary.json']],
        scope=config['scope']+' Paired descriptive contrasts only; no confidence interval, independent seed claim, or selection of successful tasks.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=summarize(args.run); args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.output/'cells.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['cells'][0])); writer.writeheader(); writer.writerows(result['cells'])
    print(json.dumps({k:v for k,v in result.items() if k not in ['comparisons','cells','inputs']}))


if __name__=='__main__':main()
