import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/submission_release_manifest.json'
TEXT_EXTENSIONS = {'.py', '.tex', '.bib', '.bst', '.sty', '.json', '.csv', '.md', '.txt'}


def encode_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def sanitize_metadata(value):
    if isinstance(value, dict):
        return {key: sanitize_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_metadata(item) for item in value]
    if not isinstance(value, str):
        return value
    windows = PureWindowsPath(value)
    if windows.is_absolute():
        normalized = windows.as_posix()
        project = ROOT.as_posix().rstrip('/') + '/'
        if normalized.lower().startswith(project.lower()):
            return normalized[len(project):]
        parts = windows.parts
        if len(parts) > 2 and parts[1] == 'CCAD_Storage':
            return 'external_assets/' + '/'.join(parts[2:])
        return 'external_assets/' + windows.name
    return value


def audit_text(name, content):
    if Path(name).suffix.lower() not in TEXT_EXTENSIONS:
        return
    value = content.decode('utf-8')
    forbidden = [r'[A-Za-z]:[\\/]Users[\\/]', r'github\.com/zz0209',
                 r'git@github\.com:zz0209', r'BEGIN (?:RSA |OPENSSH )?PRIVATE KEY']
    for pattern in forbidden:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValueError(f'Identifying or private content remains in {name}')


def collect(manifest):
    files = {}
    provenance = {}
    for name in manifest['files']:
        source = ROOT / name
        files[name] = source.read_bytes()
    for name in manifest['json_data']:
        source = ROOT / name
        original = source.read_bytes()
        value = json.loads(original)
        if name.endswith('/full_part_checks.json'):
            value = {key: value[key] for key in ['written_at_utc', 'scope', 'human']}
        files[name] = encode_json(sanitize_metadata(value))
        provenance[name] = {'original_sha256': hashlib.sha256(original).hexdigest(),
                            'transformation': 'Machine paths made relative; full_part_checks retains its human study.'}
    for name in manifest['configs']:
        original = (ROOT / name).read_bytes()
        value = sanitize_metadata(json.loads(original))
        value['source_revision'] = 'anonymous-source-bundle'
        value['environment_spec'] = 'configs/submission_environment.json'
        files[name] = encode_json(value)
        provenance[name] = {'original_sha256': hashlib.sha256(original).hexdigest(),
                            'transformation': 'Machine paths made relative; explicit source and environment identities added.'}
    files['configs/submission_environment.json'] = (ROOT / 'configs/submission_environment.json').read_bytes()
    files['README.md'] = (ROOT / 'paper/submission_README.md').read_bytes()
    files['paper/submission_assets.md'] = (ROOT / 'paper/submission_assets.md').read_bytes()
    files['CLAIMS.json'] = encode_json(manifest['claims'])
    raw_index = {'studies': []}
    for study in manifest['raw_grammar']:
        analysis = json.loads((ROOT / study['analysis']).read_text(encoding='utf-8'))
        record = {'name': study['name'], 'analysis': study['analysis'], 'runs': []}
        for item in analysis['inputs']:
            directory = Path(item['run'])
            edge = f"s{item.get('source_seed', 1)}_t{item['target_seed']}"
            prefix = f"paper/data/raw/{study['name']}/{edge}"
            responses = (directory / 'responses.npz').read_bytes()
            actual_hash = hashlib.sha256(responses).hexdigest()
            if actual_hash != item['response_sha256']:
                raise ValueError(f'Raw response identity changed for {study["name"]}/{edge}')
            panel = json.loads((directory / 'panel.json').read_text(encoding='utf-8'))
            metadata = {'query_order': panel['query_order'], 'queries': panel['queries'],
                        'rows': [{'task': row['task'], 'row_id': row['row_id']} for row in panel['rows']]}
            files[prefix + '/responses.npz'] = responses
            files[prefix + '/panel.json'] = encode_json(metadata)
            record['runs'].append({'responses': prefix + '/responses.npz', 'panel': prefix + '/panel.json',
                                   'source_seed': item.get('source_seed', 1), 'target_seed': item['target_seed'],
                                   'response_sha256': actual_hash})
        raw_index['studies'].append(record)
    files['paper/data/raw/INDEX.json'] = encode_json(raw_index)
    for name, content in files.items():
        audit_text(name, content)
    return files, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'paper/delivery')
    parser.add_argument('--validation-stage', type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    common, provenance = collect(manifest)
    if args.validation_stage:
        args.validation_stage.mkdir(parents=True, exist_ok=False)
        for name, content in common.items():
            path = args.validation_stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print(json.dumps({'validation_stage': str(args.validation_stage), 'files': len(common)}))
    if args.check:
        print(json.dumps({'checked_common_files': len(common), 'bytes': sum(map(len, common.values())),
                          'raw_grammar_runs': 16, 'formats': list(manifest['formats'])}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for format_name, spec in manifest['formats'].items():
        target = args.output / f'{format_name}_{manifest["release"]}.zip'
        if target.exists():
            raise FileExistsError(target)
        files = dict(common)
        for name in [spec['entry'], spec['pdf'], *spec['styles']]:
            files[name] = (ROOT / name).read_bytes()
            audit_text(name, files[name])
        records = [{'path': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
                   for name, content in sorted(files.items())]
        files['PACKAGE_MANIFEST.json'] = encode_json({'format': format_name,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'files': records,
            'derived_metadata': provenance, 'latex_dependency_discovery': 'Explicit reviewed allowlist'})
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, content in sorted(files.items()):
                archive.writestr(name, content)
        with zipfile.ZipFile(target) as archive:
            failure = archive.testzip()
            if failure is not None:
                raise ValueError(f'ZIP integrity failed: {failure}')
        results.append({'path': target.name, 'bytes': target.stat().st_size,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'files': len(files)})
    (args.output / 'README.md').write_bytes(common['README.md'])
    (args.output / 'PACKAGE_RESULTS.json').write_bytes(encode_json(results))
    print(json.dumps(results))


if __name__ == '__main__':
    main()
