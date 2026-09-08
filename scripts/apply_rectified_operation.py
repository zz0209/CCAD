"""Apply one stored toy source operation to target codes without Torch/SciPy."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ccad.rectified_operation import RectifiedOperation

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--collection',type=Path,required=True)
    ap.add_argument('--source',type=int,required=True);ap.add_argument('--target',type=int,required=True);ap.add_argument('--factor',type=int,required=True)
    ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--consumer',choices=['removal','contrast'],default='removal');ap.add_argument('--selected',action='store_true');ap.add_argument('--components',action='store_true')
    a=ap.parse_args();collection=json.loads(a.collection.read_text());matches=[r for r in collection['operations'] if (r['source_seed'],r['target_seed'],r['factor'])==(a.source,a.target,a.factor)]
    if len(matches)!=1:raise ValueError('Exactly one matching stored operation required')
    record=matches[0];operation=RectifiedOperation.from_record(record)
    with np.load(a.input,allow_pickle=False) as data:
        result=operation.apply(data['recipient_codes'],data['donor_codes'] if a.consumer=='contrast' else None,data['source_mask'] if 'source_mask' in data else None,a.selected,a.components)
        if a.selected and 'target_members' in data and not np.array_equal(data['target_members'],operation.target_members):raise ValueError('Stored input member IDs differ from operation')
    with a.output.open('xb') as f:np.savez_compressed(f,**result,target_members=operation.target_members,source_members=operation.source_members)
    print(json.dumps(dict(output=str(a.output),rows=len(result['delta']),consumer=a.consumer,target_members=len(operation.target_members),scope='Predicted source operation at the toy hook, not native target deletion or latent-factor truth.')))
if __name__=='__main__':main()
