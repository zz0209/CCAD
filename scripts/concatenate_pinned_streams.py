"""Restore an exact two-part training stream from an existing combined manifest.

No model, tokenizer, network or NumPy is required. Both input hashes and the
combined hash must match the preserved manifest; the output is never replaced.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--prefix',type=Path,required=True)
    ap.add_argument('--suffix',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();m=json.loads(args.manifest.read_text())
    if args.output.exists():raise FileExistsError('Use a new output path')
    for p,key in [(args.prefix,'old_prefix_sha256'),(args.suffix,'fresh_suffix_sha256')]:
        if digest(p)!=m[key]:raise ValueError('Input stream hash differs: '+str(p))
        if p.stat().st_size%2:raise ValueError('Expected uint16 stream')
    if sum(p.stat().st_size for p in [args.prefix,args.suffix])!=2*m['tokens']:
        raise ValueError('Combined token count differs')
    args.output.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
    with args.output.open('xb') as output:
        for p in [args.prefix,args.suffix]:
            with p.open('rb') as source:
                while chunk:=source.read(1<<20):output.write(chunk);h.update(chunk)
    if h.hexdigest()!=m['combined_sha256']:raise ValueError('Combined hash differs; failed output is retained')
    print(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),output=str(args.output.resolve()),
        bytes=args.output.stat().st_size,tokens=m['tokens'],sha256=h.hexdigest(),
        scope='Exact local stream concatenation; no corpus acquisition, training or scientific experiment replay.')))


if __name__=='__main__':main()
