import argparse
import json
import time
from pathlib import Path

DESCRIPTION = ('P3: write status.json for one stage in the format the CCAD progress viewer reads '
               '(tmp/progress_viewer/server.py): RUNNING with its start time, then PASS or FAIL with the end time.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--dir', type=Path, required=True)
    parser.add_argument('--status', choices=['RUNNING', 'PASS', 'FAIL'], required=True)
    args = parser.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)
    path = args.dir / 'status.json'
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    st = json.loads(path.read_text()) if path.exists() and args.status != 'RUNNING' else {}
    st['status'] = args.status
    if args.status == 'RUNNING':
        st['started_at_utc'] = now
    else:
        st['ended_at_utc'] = now
    path.write_text(json.dumps(st))


if __name__ == '__main__':
    main()
