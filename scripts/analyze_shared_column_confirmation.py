from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

from analyze_source_column_transfer import human, infinitive


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/final_science_20260921_round04'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('setting', choices=['human', 'infinitive'])
    args = parser.parse_args()
    freeze = json.loads((OUT/f'SHARED_COLUMN_{args.setting.upper()}_CONFIRMATION_FREEZE.json').read_text())
    for path, digest in freeze['identities'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    path = OUT/f'SHARED_COLUMN_{args.setting.upper()}_CONFIRMATION_ANALYSIS.json'
    if path.exists():
        raise FileExistsError(path)
    result = human(True) if args.setting == 'human' else infinitive(True)
    result.update(written_at_utc=datetime.now(timezone.utc).isoformat(), freeze_verified=True,
                  fit_targets=freeze['fit_targets'], held_targets=freeze['held_targets'])
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({m: v['held_requests'] for m, v in result['summary'].items()}, indent=2))
    print(json.dumps([d for d in result['differences'] if d['family'] == 'held_requests'], indent=2))


if __name__ == '__main__':
    main()
