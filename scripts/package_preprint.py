"""Prepare a local preprint source archive and its retained-evidence companion.

This copies project material only. It never uploads, downloads or runs models.
The source archive contains the LaTeX dependency closure. The companion keeps
current-claim evidence, executable snapshots and acquisition records; model
weights and large activation caches remain separately acquired inputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'


def identity(path):
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=digest)


def paper_closure():
    found = set()

    def visit(path):
        path = path.resolve()
        if not path.is_relative_to(PAPER) or not path.is_file():
            raise ValueError(f'Missing or non-paper dependency: {path}')
        if path in found:
            return
        found.add(path)
        if path.suffix != '.tex':
            return
        text = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text(encoding='utf-8'))
        for rel in re.findall(r'\\(?:input|tableinput)\{([^}]+)\}', text):
            p = PAPER / rel
            visit(p if p.suffix else p.with_suffix('.tex'))
        for rel in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
            p = PAPER / rel
            visit(p if p.suffix else p.with_suffix('.pdf'))

    visit(PAPER / 'main.tex')
    found.update([PAPER / 'references.bib', PAPER / 'main.bbl'])
    assert all(p.is_file() for p in found)
    return found


def plan():
    source = paper_closure()
    files = set(source)
    omitted, missing = {}, set()
    index = json.loads((PAPER / 'EVIDENCE_INDEX.json').read_text(encoding='utf-8'))
    claims = [c for c in index['claims'] if c.get('current_manuscript')]
    runs = set()

    def add(path):
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            return
        if not path.is_file():
            missing.add(path.relative_to(ROOT).as_posix())
            return
        rel = path.relative_to(ROOT)
        if path.suffix in {'.pt', '.pth', '.safetensors', '.bin', '.npy'} or (
                path.suffix == '.npz' and path.stat().st_size > 32_000_000):
            omitted[rel.as_posix()] = dict(bytes=path.stat().st_size,
                recovery='Acquire the pinned input or replay its producer; use config/inputs and source_snapshot. Original local file retained.')
        else:
            files.add(path)

    def refs(value):
        if isinstance(value, dict):
            for k, v in value.items():
                if k == 'path' and isinstance(v, str):
                    rel = v.replace('\\', '/')
                    if rel.startswith(('runs/', 'artifacts/', 'paper/', 'configs/', 'scripts/', 'src/')):
                        add(ROOT / rel)
                        if rel.startswith('runs/'):
                            runs.add(rel.split('/')[1])
                refs(v)
        elif isinstance(value, list):
            for v in value:
                refs(v)

    refs(claims)
    refs(index.get('current_source_version_note', {}))
    for p in (ROOT / 'artifacts/final_wrap_20260915').iterdir():
        if p.is_file() and p.suffix in {'.md', '.json'}:
            add(p)
    # Companion material preserves run inputs and original source versions, even
    # when a claim index records only its derived result table.
    for c in claims:
        for run in re.findall(r'runs[/\\]([^/\\"\s]+)', json.dumps(c)):
            runs.add(run)
    # The headline raw-output replays use these fixed panels and source banks.
    runs.update(['REFORM_R57_binding_member_selection_v1_20260915',
                 'REFORM_R56_binding_roles_confirm_s1_v1_20260915',
                 'REFORM_R39_qwen_member_confirmation_v1_20260914'])
    runs.update(f'REFORM_R57_binding_member_confirm_t{s}_v1_20260915' for s in range(2, 6))
    runs.update(f'REFORM_R59_shift_confirm_seed{s}_v1_20260915' for s in range(1, 6))
    for run in sorted(runs):
        folder = ROOT / 'runs' / run
        if not folder.is_dir():
            missing.add('runs/' + run)
            continue
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc', '.pyo'}:
                add(p)
    for folder in ['src', 'scripts', 'configs', 'tests', '.aris/compute']:
        for p in (ROOT / folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py', '.json', '.md', '.toml', '.txt', '.yaml', '.yml'}:
                add(p)
    for p in (ROOT / 'artifacts/correspondence_reform_20260913').iterdir():
        if p.is_file() and p.suffix in {'.md', '.json', '.npz'}:
            add(p)
    for p in (PAPER / 'data').iterdir():
        if p.is_file():
            add(p)
    for p in list(source):
        if p.suffix == '.pdf':
            for ext in ['.svg', '.png']:
                if p.with_suffix(ext).exists():
                    add(p.with_suffix(ext))
    for rel in ['paper/main.pdf', 'paper/README.md', 'paper/CLAIM_MAP.md', 'paper/EVIDENCE_INDEX.json',
                'paper/figures/FIGURE_MANIFEST.json', 'AGENTS.md', 'EXPERIMENT_PLAN.md',
                'EXPERIMENT_TRACKER.md', 'REFERENCE_REGISTRY.md', 'RUN_ARTIFACT_CONTRACT.md', 'master_log.md']:
        add(ROOT / rel)
    for p in (ROOT / 'delivery').iterdir():
        if p.is_file() and p.suffix in {'.md', '.json', '.jsonl'}:
            add(p)
    return source, files, omitted, missing, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source, files, omitted, missing, runs = plan()
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_files=len(source), companion_files=len(files),
                  source_bytes=sum(p.stat().st_size for p in source),
                  companion_bytes=sum(p.stat().st_size for p in files),
                  run_directories=sorted(runs), omitted_local_inputs=omitted,
                  missing_references=sorted(missing),
                  scope='Local human-review package; original model/data licenses apply. No publication performed.')
    (output / 'PACKAGE_PLAN.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['source_files', 'companion_files', 'source_bytes', 'companion_bytes', 'missing_references']}), flush=True)
    if not args.build:
        return
    if missing:
        raise ValueError('Resolve the listed missing current-evidence references before packaging')
    receipts = []
    for name, paths, base in [('arxiv_source.zip', source, PAPER), ('research_companion.zip', files, ROOT)]:
        target = output / name
        entries = []
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in sorted(paths):
                before = p.stat()
                item = identity(p)
                z.write(p, p.relative_to(base).as_posix())
                after = p.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f'Input changed during packaging: {p}')
                entries.append(item)
            if name == 'research_companion.zip':
                z.writestr('PACKAGE_CONTENTS.json', json.dumps(dict(files=entries, **result), indent=2))
        with zipfile.ZipFile(target) as z:
            assert z.testzip() is None
        manifest = output / (name + '.manifest.json')
        manifest.write_text(json.dumps(dict(archive=identity(target), files=entries), indent=2) + '\n', encoding='utf-8')
        receipts.append(identity(target))
        print(json.dumps(receipts[-1]), flush=True)
    (output / 'PACKAGE_RECEIPT.json').write_text(json.dumps(receipts, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
