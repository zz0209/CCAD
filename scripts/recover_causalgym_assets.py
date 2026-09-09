"""Recover pinned public benchmark files omitted from the local delivery.

Downloads data and the original Pair/Batch module, verifies retained hashes,
and records retrieval. It never runs downloaded code. Test retrieval must be
explicit, and requires a previously written freeze for the proposed execution.
"""
from pathlib import Path
import argparse,json,hashlib,urllib.request
from datetime import datetime,timezone
DATA_REV='95349c3a5e53e2506e8b212482ea6dd784978156'
CODE_REV='0f3129ff3b6c5c8264892f30a25be25150ae9179'
HASHES={'train.json':'021f40afe202c312db3523497988097f77deb59d74c487234f28b7d3360ad0ba','dev.json':'0f32cdfa742e7633bb13e49601d346f832f9c17b19107f1eeaffd53995912ce8','test.json':'d433abfc2b910f9cc339cc0be392a85b9f3fa4361e07e235816ade821144cdef','data.py':'136da30d820bfc3cd64866554e9d5e6ee4fb53f504255c4196549ef8aec23939'}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--include-test',action='store_true');p.add_argument('--freeze',type=Path);a=p.parse_args()
 if a.include_test and (a.freeze is None or not a.freeze.is_file()):raise ValueError('Provide the prospective execution freeze before requesting test')
 a.output.mkdir(parents=True,exist_ok=True);records=[]
 names=['train.json','dev.json','data.py']+(['test.json'] if a.include_test else [])
 for name in names:
  url=(f'https://raw.githubusercontent.com/aryamanarora/causalgym/{CODE_REV}/data.py' if name=='data.py' else f'https://huggingface.co/datasets/aryaman/causalgym/resolve/{DATA_REV}/{name}')
  path=a.output/name;retrieved=False
  if not path.exists():
   with urllib.request.urlopen(url,timeout=60) as r:content=r.read()
   actual=hashlib.sha256(content).hexdigest()
   if actual!=HASHES[name]:raise ValueError('Remote bytes differ from retained identity: '+name)
   with path.open('xb') as f:f.write(content)
   retrieved=True
  actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==HASHES[name],('Existing local identity differs; file preserved',path)
  records.append(dict(path=str(path.resolve()),url=url,sha256=actual,retrieved=retrieved,scope='Original code module is acquired for execution from its author repository, not relicensed by this package.' if name=='data.py' else 'Pinned original dataset records; no parsing or sampling by this helper.'))
 receipt=dict(written_at_utc=datetime.now(timezone.utc).isoformat(),files=records,freeze_sha256=hashlib.sha256(a.freeze.read_bytes()).hexdigest() if a.freeze else None,scope='Public immutable acquisition and hash verification only; neither scientific evaluation nor permission to redistribute third-party code.')
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');(a.output/f'RETRIEVAL_{stamp}.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
