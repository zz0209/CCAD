"""Extract exactly the published 55 ReLU features; retain the original archive."""
from pathlib import Path
import io,json,hashlib,datetime,zipfile,time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913/r57_shift_consumer'
BULK=Path('D:/CCAD_Storage/references/feature_circuits_shift/saes/50a434461d36ed78d1b0b901944e6edc829f1dce')


def main():
    torch.set_num_threads(2);start=time.perf_counter();torch.manual_seed(5701)
    human=json.loads((ART/'PUBLISHED_EXPLANATION.json').read_text());out={};inputs=[]
    with zipfile.ZipFile(BULK/'dictionaries_pythia-70m-deduped_10.zip') as archive:
        for site,ids in human['members'].items():
            folder='embed' if site=='embed' else site.split('_')[0]+'_out_layer'+site.split('_')[1]
            member=f'dictionaries/pythia-70m-deduped/{folder}/10_32768/ae.pt'
            raw=archive.read(member);state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
            assert set(state)=={'bias','encoder.weight','encoder.bias','decoder.weight'}
            x=torch.randn(16,512);full=torch.relu(torch.nn.functional.linear(x-state['bias'],state['encoder.weight'],state['encoder.bias']))[:,ids]
            enc=state['encoder.weight'][ids].clone();eb=state['encoder.bias'][ids].clone();dec=state['decoder.weight'][:,ids].T.clone()
            partial=torch.relu(torch.nn.functional.linear(x-state['bias'],enc,eb));error=float((partial-full).abs().max())
            assert torch.allclose(partial,full,atol=2e-5,rtol=2e-5),error
            for key,value in [('ids',np.array(ids,dtype=np.int64)),('encoder',enc.numpy()),('encoder_bias',eb.numpy()),('decoder',dec.numpy()),('center',state['bias'].numpy())]:out[site+'__'+key]=value
            inputs.append(dict(site=site,zip_member=member,sha256=hashlib.sha256(raw).hexdigest(),members=ids,relu_extraction_max_error=error,config=json.loads(archive.read(member.replace('ae.pt','config.json')))))
    dest=BULK/'published_shift55.npz';np.savez_compressed(dest,**out)
    record=dict(written_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.perf_counter()-start,path=str(dest),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),inputs=inputs,
        operation='At each original module output x, compute selected ReLU features z_I=relu(W_E[I](x-b_D)+b_E[I]) and apply x-D_I z_I. Exact selected coordinates of the published full ReLU SAE; original reconstruction residual retained.',
        scope='Source artifact preparation and algebraic encoding comparison on random inputs. No language-model, classifier, fairness or correspondence result yet.',
        license='MIT; upstream copyright and LICENSE retained next to the source-reading manifest. Source member choices are the published author annotations, not new human or AI annotations.')
    (ART/'EXTRACTION_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='inputs'}))


if __name__=='__main__':main()
