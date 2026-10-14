import argparse
import json
from pathlib import Path

DESCRIPTION = ('Inventory of downloaded assets: for every top-level folder of the assets directory, its files with sizes, '
               'and every manifest JSON at the top level copied verbatim, so that each folder can be downloaded again '
               'from the repository and revision its manifest records. Read-only.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    folders = {}
    for d in sorted(p for p in args.assets.iterdir() if p.is_dir()):
        files = [dict(path=str(f.relative_to(d)), bytes=f.stat().st_size) for f in sorted(d.rglob('*')) if f.is_file()]
        folders[d.name] = dict(bytes=sum(f['bytes'] for f in files), files=files)
    manifests = {p.name: json.loads(p.read_text()) for p in sorted(args.assets.glob('*.json'))}
    nested = {str(p.relative_to(args.assets)): json.loads(p.read_text())
              for p in sorted(args.assets.glob('*/*MANIFEST*.json'))}
    args.out.write_text(json.dumps(dict(description=DESCRIPTION, assets=str(args.assets), folders=folders,
                                        manifests=manifests, nested_manifests=nested), indent=1))
    for name, f in folders.items():
        print(f"{f['bytes'] / 2**30:8.2f} GiB  {len(f['files']):6d} files  {name}")


if __name__ == '__main__':
    main()
