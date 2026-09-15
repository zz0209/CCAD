"""Fetch pinned public SHIFT assets without running external notebook code."""
from pathlib import Path
import argparse,ast,datetime,hashlib,json,time,urllib.request

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/correspondence_reform_20260913/r57_shift_consumer'
DEST=Path('D:/CCAD_Storage/references/feature_circuits_shift')


def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');a=p.parse_args()
    notebook=json.loads((ART/'bib_shift.ipynb').read_text())
    source=''.join(notebook['cells'][11]['source']);node=ast.parse(source).body[0].value
    assert isinstance(node,ast.Dict)
    masks={int(k.slice.value):ast.literal_eval(v) for k,v in zip(node.keys,node.values)}
    sites=['embed']+[f'{kind}_{layer}' for layer in range(5) for kind in ['attn','mlp','resid']]
    selected={sites[i]:v for i,v in masks.items() if v}
    annotation=dict(source_commit='cf080789e16f50238097db1fb9a3947a6cfcf9f6',notebook_cell=11,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),members=selected,total_members=sum(map(len,selected.values())),
        meaning='Published author-selected gender-related features to ablate while preserving profession classification. Comments in the source retain interpretations and intentionally excluded profession-related members.',
        model='EleutherAI/pythia-70m-deduped',classifier_layer=4,dictionary_id=10,dictionary_width=32768)
    (ART/'PUBLISHED_EXPLANATION.json').write_text(json.dumps(annotation,indent=2)+'\n')
    plan=[]
    for kind,repo,meta in [('models','saprmarks/pythia-70m-deduped-saes','saes_api'),('datasets','LabHC/bias_in_bios','bios_api')]:
        api=json.loads((ART/(meta+'.json')).read_text());revision=api['sha']
        for f in api['siblings']:
            name=f['rfilename']
            if not(name.endswith('.zip') or name.endswith('.parquet') or name=='README.md'):continue
            prefix='datasets/' if kind=='datasets' else ''
            url=f'https://huggingface.co/{prefix}{repo}/resolve/{revision}/{name}'
            dest=DEST/('saes' if kind=='models' else 'bios')/revision/name
            plan.append(dict(url=url,path=str(dest),expected_bytes=f['size'],expected_sha256=f.get('lfs',{}).get('sha256')))
    (ART/'ASSET_PLAN.json').write_text(json.dumps(dict(files=plan,total_bytes=sum(f['expected_bytes'] for f in plan)),indent=2)+'\n')
    if not a.download:
        print(json.dumps(dict(member_count=annotation['total_members'],sites=len(selected),download_bytes=sum(f['expected_bytes'] for f in plan))));return
    receipts=[]
    for f in plan:
        dest=Path(f['path']);dest.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
        if dest.exists():
            with dest.open('rb') as stream:hashval=hashlib.file_digest(stream,'sha256').hexdigest()
            assert dest.stat().st_size==f['expected_bytes'] and (f['expected_sha256'] is None or hashval==f['expected_sha256'])
        else:
            part=dest.with_suffix(dest.suffix+'.partial');digest=hashlib.sha256();size=0;last=time.perf_counter()
            with urllib.request.urlopen(f['url'],timeout=60) as response,part.open('wb') as stream:
                while data:=response.read(8*1024*1024):
                    stream.write(data);digest.update(data);size+=len(data)
                    if time.perf_counter()-last>20:
                        print(json.dumps(dict(file=dest.name,bytes=size,total=f['expected_bytes'],seconds=time.perf_counter()-start)),flush=True);last=time.perf_counter()
            hashval=digest.hexdigest();assert size==f['expected_bytes'] and (f['expected_sha256'] is None or hashval==f['expected_sha256'])
            part.replace(dest)
        receipts.append(dict(f,sha256=hashval,seconds=time.perf_counter()-start,completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        (ART/'ASSET_RECEIPT.json').write_text(json.dumps(receipts,indent=2)+'\n');print(json.dumps(receipts[-1]),flush=True)


if __name__=='__main__':main()
