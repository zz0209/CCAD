"""Supplementary analysis of already saved, post-adaptation readout outputs.

The frozen primary analysis is retained unchanged. This adds every saved
source-direction execution after dictionary adaptation, with its paired draws.
"""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_science04_confirmation import OUT, BULK, families, weights, summarize

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bulk-root',type=Path,default=BULK)
    p.add_argument('--output',type=Path,default=OUT/'ROUND04_INFINITIVE_READOUT_SUPPLEMENT.json')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    seeds=[1,3,4,5]
    runs=[args.bulk_root/f'SCIENCE04_infinitive_t{s}_v{2 if s==1 else 1}_20260919' for s in seeds]
    idx=json.loads((runs[0]/'INDEX.json').read_text())
    rows=idx['rows'];names=idx['queries']
    panel=json.loads((OUT/'ROUND04_INFINITIVE_PANEL.json').read_text())
    ff=families(panel['families'],names)
    values=[dict(np.load(r/'responses.npz')) for r in runs]
    source=values[0]['source'].astype(float);clean=values[0]['none'].astype(float)
    assert all(np.array_equal(v['source'],source) and np.array_equal(v['none'],clean) for v in values)
    methods=['tangent_gain','tangent_mixed','raw_reconstruction',
             'raw_reconstruction_after_tangent_gain','raw_reconstruction_after_tangent_mixed']
    nums={m:np.stack([((v[m].astype(float)-source)**2)[None] for v in values]) for m in methods}
    den=((source-clean)**2)[None]
    verbs=sorted({r['verb'] for r in rows});nouns=sorted({r['noun'] for r in rows})
    vi=np.array([verbs.index(r['verb']) for r in rows]);ni=np.array([nouns.index(r['noun']) for r in rows])
    rng=np.random.default_rng(2026091942)
    dw=[weights(rng,len(verbs))[:,vi]*weights(rng,len(nouns))[:,ni]]
    result=summarize(methods,nums,den,dw,ff,rng,np.sqrt(den.mean(-1)).tolist(),seeds,
                     dict(setting='infinitive',scope='Supplementary analysis of retained outputs; primary unchanged',runs=list(map(str,runs))))
    primary=json.loads((OUT/'ROUND04_INFINITIVE_ANALYSIS.json').read_text())
    for m in ['tangent_gain','tangent_mixed','raw_reconstruction']:
        assert result['summary'][m]==primary['summary'][m]
    result['quality']={str(s):json.loads((r/'quality.json').read_text()) for s,r in zip(seeds,runs)}
    dest=args.output
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print({m:v['held_requests'] for m,v in result['summary'].items()})

if __name__=='__main__':main()
