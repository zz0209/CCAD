from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

import transformers

from naacl_qwen_prepare import TASKS, retokenize

DESCRIPTION = ('Retokenize a further BLiMP confirmation panel for the Qwen3 grammar study and verify that its pairs are '
               'disjoint from the fitting and earlier confirmation panels.')


def pairs(path):
    return {(r['sentence_good'], r['sentence_bad']) for r in json.loads(Path(path).read_text())['rows']}


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--exclude', type=Path, nargs='+', required=True)
    parser.add_argument('--prefix', default='\n')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = [r for r in json.loads(args.panel.read_text())['rows'] if r['task'] in TASKS]
    fresh = {(r['sentence_good'], r['sentence_bad']) for r in rows}
    overlaps = {str(p): len(fresh & pairs(p)) for p in args.exclude}
    assert not any(overlaps.values()), overlaps
    tok = transformers.AutoTokenizer.from_pretrained(args.model_dir)
    out = retokenize(rows, tok, args.prefix)
    scope = 'Further BLiMP confirmation pairs from the GPT-2 cross-source study, never used in the Qwen3 study before this panel.'
    (args.output_dir / 'confirmation_panel.json').write_text(json.dumps(dict(rows=out, scope=scope)))
    record = dict(written_utc=datetime.now(timezone.utc).isoformat(), panel=str(args.panel),
                  panel_sha256=hashlib.sha256(args.panel.read_bytes()).hexdigest(), rows=len(out),
                  by_task={t: sum(r['task'] == t for r in out) for t in TASKS}, overlaps=overlaps)
    (args.output_dir / 'FRESH_PANEL.json').write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
