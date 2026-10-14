import argparse
import hashlib
from multiprocessing import Pool
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='SHA256 and size of every regular file under the given folders of a '
                                                 'base directory, minus excluded prefixes, as tab-separated lines.')
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--roots', nargs='+', required=True)
    parser.add_argument('--exclude', nargs='*', default=[])
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    files = []
    for root in args.roots:
        for p in sorted((args.base / root).rglob('*')):
            rel = p.relative_to(args.base).as_posix()
            if p.is_file() and not p.is_symlink() and not any(rel.startswith(e) for e in args.exclude):
                files.append(p)
    with Pool(args.workers) as pool:
        sums = pool.map(digest, files, chunksize=8)
    with open(args.out, 'w') as out:
        for p, s in zip(files, sums):
            out.write(f'{s}\t{p.stat().st_size}\t{p.relative_to(args.base).as_posix()}\n')
    print(len(files), sum(p.stat().st_size for p in files))


if __name__ == '__main__':
    main()
