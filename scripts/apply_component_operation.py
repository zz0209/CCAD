"""Apply saved source masks to target codes, returning a residual update."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from ccad.component_operation import ComponentOperation


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['operation','input','output']:ap.add_argument('--'+name,required=True,type=Path)
    ap.add_argument('--consumer',choices=['contribution','removal','contrast'],default='removal')
    ap.add_argument('--dose',type=float,default=1.)
    ap.add_argument('--scales-key',default='source_scales',help='Array key in input; absent means whole source group')
    args=ap.parse_args();op=ComponentOperation.load(args.operation)
    with np.load(args.input,allow_pickle=False) as data:
        if not np.array_equal(data['target_members'],op.target_members):raise ValueError('Target member IDs/order differ')
        scales=data[args.scales_key] if args.scales_key in data else None
        if scales is not None and not np.array_equal(data['source_members'],op.source_members):raise ValueError('Source mask IDs/order differ')
        delta=op.apply(data['selected_codes'],scales,args.consumer,data['selected_donor_codes'] if args.consumer=='contrast' else None,args.dose)
        result=dict(delta=delta,source_members=op.source_members,target_members=op.target_members,metadata_json=np.array(json.dumps(op.metadata)))
        if 'row_ids' in data:result['row_ids']=data['row_ids']
        with args.output.open('xb') as f:np.savez_compressed(f,**result)
    print(json.dumps(dict(output=str(args.output.resolve()),shape=list(delta.shape),consumer=args.consumer,dose=args.dose,scope='Source-aligned residual update; not target-native or semantic deletion. New masks/doses outside declared validation remain untested.')))


if __name__=='__main__':main()
