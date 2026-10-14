import argparse
import json
import os
from pathlib import Path

DESCRIPTION = ('Inventory of a Hugging Face hub cache: for every repository, the bytes of the files its snapshots point '
               'to (symlink targets resolved, each file counted once per repository), the files of each snapshot, and '
               'the files in blob stores that no snapshot points to. Read-only.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--hub', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    referenced, repos = set(), {}
    for repo in sorted(args.hub.glob('*--*--*')):
        targets, files = {}, []
        for link in (repo / 'snapshots').rglob('*'):
            if link.is_symlink() or link.is_file():
                real = Path(os.path.realpath(link))
                if real.is_file():
                    targets[str(real)] = real.stat().st_size
                    files.append(dict(path=str(link.relative_to(repo / 'snapshots')), bytes=real.stat().st_size))
        referenced |= set(targets)
        repos[repo.name] = dict(bytes=sum(targets.values()), files=len(files),
                                revisions=sorted(p.name for p in (repo / 'snapshots').iterdir()) if (repo / 'snapshots').is_dir() else [],
                                file_list=sorted(files, key=lambda f: -f['bytes']))
    orphans = []
    for store in [args.hub / 'blobs'] + [r / 'blobs' for r in args.hub.glob('*--*--*')]:
        if store.is_dir():
            for f in store.rglob('*'):
                if f.is_file() and not f.is_symlink() and str(f.resolve()) not in referenced:
                    orphans.append(dict(path=str(f), bytes=f.stat().st_size))
    out = dict(description=DESCRIPTION, hub=str(args.hub),
               repos=dict(sorted(repos.items(), key=lambda kv: -kv[1]['bytes'])),
               orphan_bytes=sum(o['bytes'] for o in orphans), orphans=sorted(orphans, key=lambda o: -o['bytes'])[:200])
    args.out.write_text(json.dumps(out, indent=1))
    for name, r in out['repos'].items():
        print(f"{r['bytes'] / 2**30:8.1f} GiB  {r['files']:5d} files  {name}")
    print(f"{out['orphan_bytes'] / 2**30:8.1f} GiB  {len(orphans)} files not referenced by any snapshot")


if __name__ == '__main__':
    main()
