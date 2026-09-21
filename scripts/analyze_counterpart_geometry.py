from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np
import torch


def basis(decoder, members):
    return torch.linalg.qr(decoder[:,members].double(),mode='reduced')[0]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--banks',type=Path,nargs='+',required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2)
    c=json.loads(a.config.read_text());covariances=dict(np.load(a.source/'source_action_bank.npz'))
    rows=[];identities={str(a.source/'source_action_bank.npz'):hashlib.sha256((a.source/'source_action_bank.npz').read_bytes()).hexdigest()}
    for path in a.banks:
        bank=json.loads(path.read_text());seed=bank['target_seed'];parts=bank['supports']['action_span_64']
        identities[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for site in parts['singular']:
            weight=Path(c['target_directory'])/f'{site}_seed{seed}.pt'
            decoder=torch.load(weight,map_location='cpu',weights_only=True)['decoder.weight']
            identities[str(weight)]=hashlib.sha256(weight.read_bytes()).hexdigest()
            bases={part:basis(decoder,ids[site]) for part,ids in parts.items() if ids[site]}
            for part in ['singular','plural']:
                g=torch.from_numpy(covariances[site+'__'+part]).double();energy=float(g.trace())
                if energy<=1e-20:continue
                other='plural' if part=='singular' else 'singular'
                exchange=len(parts[part][site])==len(parts[other][site])
                own=bases[part];opposite=bases[other] if exchange else own
                rows.append(dict(target=seed,site=site,part=part,exchanged=exchange,
                    own_capture=float(torch.trace(own.T@g@own)/energy),
                    exchanged_capture=float(torch.trace(opposite.T@g@opposite)/energy),
                    span_overlap=float((own.T@opposite).square().sum()/own.shape[1]),
                    shared_members=len(set(parts[part][site])&set(parts[other][site]))))
    output=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),rows=rows,identities=identities,
        scope='Descriptive source-bank geometry. Orthogonal span capture ignores code-capacity constraints and is not a functional score.')
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    selected=[r for r in rows if r['exchanged']]
    print(json.dumps(dict(records=len(rows),exchangeable=len(selected),
        mean={k:float(np.mean([r[k] for r in selected])) for k in ['own_capture','exchanged_capture','span_overlap','shared_members']})))


if __name__=='__main__':main()
