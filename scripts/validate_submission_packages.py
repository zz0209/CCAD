import hashlib
import json
import re
import zipfile
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]


def main():
    formats = [('acl_arr', 'main', 19), ('iclr_2027', 'main_iclr', 22)]
    checks = []
    for name, entry, expected_pages in formats:
        archive_path = ROOT / f'paper/delivery/{name}_20260924.zip'
        stage = ROOT / f'paper/build/{name}_package_review_20260924'
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f'Corrupt archive: {name}')
            inventory = json.loads(archive.read('PACKAGE_MANIFEST.json'))
            for item in inventory['files']:
                payload = archive.read(item['path'])
                if len(payload) != item['bytes'] or hashlib.sha256(payload).hexdigest() != item['sha256']:
                    raise ValueError(f'Inventory mismatch: {item["path"]}')
            for filename in archive.namelist():
                parts = Path(filename).parts
                if any(part in {'.git', '.aris', 'AGENTS.md', 'MASTER_LOG.md', 'master_log.md'} for part in parts):
                    raise ValueError(f'Private project file included: {filename}')
                if Path(filename).suffix in {'.py', '.tex', '.json', '.md', '.csv', '.bib'}:
                    text = archive.read(filename).decode('utf-8')
                    if re.search(r'[A-Za-z]:[\\/]Users[\\/]|github\.com/zz0209|git@github\.com:zz0209', text, re.I):
                        raise ValueError(f'Identifying path in {filename}')
            original_pdf = ROOT / f'paper/{entry}.pdf'
            if archive.read(f'paper/{entry}.pdf') != original_pdf.read_bytes():
                raise ValueError('Distributed PDF differs from the approved manuscript')
        original = PdfReader(original_pdf)
        rebuilt = PdfReader(stage / f'paper/{entry}.pdf')
        if len(original.pages) != expected_pages or len(rebuilt.pages) != expected_pages:
            raise ValueError(f'Page count changed for {name}')
        for index, (first, second) in enumerate(zip(original.pages, rebuilt.pages)):
            if first.extract_text() != second.extract_text():
                raise ValueError(f'Page text changed for {name}, page {index + 1}')
        if original.metadata.author not in {None, '', 'Anonymous authors'}:
            raise ValueError(f'PDF author metadata remains in {name}')
        text = '\n'.join(page.extract_text() for page in original.pages)
        if re.search(r'github\.com/zz0209|[A-Za-z]:[\\/]Users[\\/]', text, re.I):
            raise ValueError(f'Identifying PDF text remains in {name}')
        log = (stage / f'paper/{entry}.log').read_text(encoding='utf-8', errors='replace')
        if 'Overfull' in log or 'undefined' in log:
            raise ValueError(f'Unresolved layout or reference warning in {name}')
        replay = json.loads((stage / 'reproduced/REPRODUCED.json').read_text())
        checks.append({'format': name, 'archive': archive_path.name,
                       'sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                       'bytes': archive_path.stat().st_size, 'files': len(inventory['files']) + 1,
                       'pages': expected_pages, 'every_page_text_identical': True,
                       'approved_pdf_bytes_identical': True, 'zip_and_file_hashes_verified': True,
                       'anonymous_file_and_text_checks': 'PASS',
                       'overfull_or_undefined_warnings': 0,
                       'compiler': 'Tectonic 0.17.0',
                       'compile_command': f'tectonic --untrusted --keep-logs --keep-intermediates --only-cached {entry}.tex',
                       'recomputed_tables': replay['tables_recomputed_from_saved_numerical_exports'],
                       'raw_grammar_recomputation': replay['raw_grammar_recomputation']})
    destination = ROOT / 'paper/delivery/VALIDATION.json'
    destination.write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    print(json.dumps([{key: value for key, value in row.items() if key != 'raw_grammar_recomputation'} for row in checks]))


if __name__ == '__main__':
    main()
