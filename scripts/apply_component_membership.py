"""Apply a saved R26 membership to actual target SAE codes, using NumPy."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map',required=True,type=Path)
    ap.add_argument('--input',required=True,type=Path,help='NPZ: target_members, selected_codes, selected_decoder_rows; optional hidden.')
    ap.add_argument('--weights',required=True,nargs=3,type=float,metavar=('VERB','NUMBER','GENDER'))
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();c=np.asarray(args.weights,dtype=np.float64)
    if not np.isfinite(c).all() or (c<0).any() or (c>1).any():raise ValueError('Component weights must lie in [0,1].')
    with np.load(args.map,allow_pickle=False) as a:members=a['target_members'];M=a['partition64'].astype(np.float64)
    if (M<0).any() or (M.sum(1)>1+2e-7).any():raise ValueError('Saved membership violates native capacity.')
    with np.load(args.input,allow_pickle=False) as a:
        if not np.array_equal(a['target_members'],members):raise ValueError('Target member IDs/order differ.')
        z=a['selected_codes'];d=a['selected_decoder_rows']
        if z.ndim!=2 or d.ndim!=2 or z.shape[1]!=len(members) or d.shape[0]!=len(members):raise ValueError('Code/decoder dimensions differ.')
        if not np.isfinite(z).all() or not np.isfinite(d).all() or (z<0).any():raise ValueError('Actual nonnegative target codes and finite decoder rows required.')
        gates=np.clip(M@c,0,1);removed=(z*gates)@d
        result=dict(delta=-removed,remaining_selected_codes=z*(1-gates),target_members=members,component_weights=c)
        if 'hidden' in a:
            h=a['hidden']
            if h.shape!=removed.shape:raise ValueError('Hidden-state dimensions differ.')
            result['edited_hidden']=h-removed
    with args.output.open('xb') as f:np.savez_compressed(f,**result)
    print(json.dumps(dict(output=str(args.output.resolve()),shape=list(removed.shape),scope='Native code deletion with original residual retained. Pure code-level execution; model continuation and functional transfer are separately evaluated.')))


if __name__=='__main__':main()
