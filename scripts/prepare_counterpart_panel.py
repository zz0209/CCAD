from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import hashlib
import json
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT/'configs/rg03_trajectory_development.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(config['model_local_dir'],local_files_only=True)
    prior = list((ROOT/'artifacts/final_science_20260920_round02').glob('AGREEMENT_*.json'))
    prior.append(ROOT/'artifacts/reuse_generalization_20260921_round03/NUMBER_CONFIRMATION_PANEL.json')
    excluded = set()
    for path in prior:
        for row in json.loads(path.read_text()).get('rows',[]):
            excluded.update(row[key] for key in ['clean_prefix','patch_prefix'] if key in row)
    manifest = ROOT/'artifacts/reuse_generalization_20260921_round03/SFC_TEST_SOURCE_MANIFEST.json'
    rows = []
    counts = {}
    for asset in json.loads(manifest.read_text(encoding='utf-8-sig')):
        path = Path(asset['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset['sha256']:
            raise ValueError('Official source changed')
        structure = path.stem.removesuffix('_test')
        if structure == 'simple':
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()]
        cells = defaultdict(dict)
        for row in records:
            clean, patch = row['clean_prefix'], row['patch_prefix']
            if clean in excluded or patch in excluded:
                continue
            if len(tokenizer.encode(clean)) != len(tokenizer.encode(patch)):
                continue
            if any(len(tokenizer.encode(row[key],add_special_tokens=False)) != 1 for key in ['clean_answer','patch_answer']):
                continue
            pair = json.dumps(sorted([clean,patch]))
            digest = hashlib.sha256(clean.encode()).hexdigest()
            cells[row['case']][clean] = dict(row,structure=structure,document_sha256=digest,
                pair_sha256=hashlib.sha256(pair.encode()).hexdigest())
        counts[structure] = {key:len(value) for key,value in cells.items()}
        for case,values in sorted(cells.items()):
            if len(values) < 8:
                raise ValueError((structure,case,len(values)))
            rows.extend(sorted(values.values(),key=lambda r:hashlib.sha256(('rg04counterpart/'+r['document_sha256']).encode()).hexdigest())[:8])
    if len(rows)!=64 or len({r['document_sha256'] for r in rows})!=64:
        raise ValueError('Unexpected confirmation size')
    out = ROOT/'artifacts/reuse_generalization_20260921_round04/NUMBER_CONFIRMATION_PANEL.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),rows=rows,
        excluded_prefixes=len(excluded),available_by_case=counts,
        exclusion_assets={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in prior},
        source_manifest=str(manifest),sampling='Eight hashed prefixes per original grammar case; exclude clean and paired prefixes from prior evaluation.'),indent=2)+'\n')
    print(json.dumps(dict(output=str(out),rows=len(rows),available=counts)))


if __name__ == '__main__':
    main()
