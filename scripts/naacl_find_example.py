from pathlib import Path
import argparse
import json

from naacl_analyze_e1_shift import load

DESCRIPTION = 'Find the first panel biography where complete-edit agreement hides a part error for the named methods.'
QUERIES = ['full', 'pronouns', 'names', 'associated_words']


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--hidden', nargs='+', default=['whole', 'hungarian_decoder'])
    parser.add_argument('--faithful', default='parts')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    docs, values, labels, genders = load(args.run)
    index = {d: i for i, d in enumerate(docs)}
    panel = json.loads(args.panel.read_text())
    split = json.loads((args.run / 'config.resolved.json').read_text())['evaluation_split']
    rows = [r for r in panel['rows'] if r['split'] == split]
    answer = lambda m, q, i: int(values[(m, q)][i] > 0)
    for r in rows:
        i = index[r['document_sha256']]
        src = [answer('source', q, i) for q in QUERIES]
        if len(set(src[1:])) < 2:
            continue
        ok = all(answer(args.faithful, q, i) == s for q, s in zip(QUERIES, src))
        hidden = all(answer(m, 'full', i) == src[0] and any(answer(m, q, i) != s for q, s in zip(QUERIES[1:], src[1:]))
                     for m in args.hidden)
        if ok and hidden:
            record = dict(document_sha256=r['document_sha256'], label=int(labels[i]), gender=int(genders[i]),
                          rule='first panel row whose source part answers differ, where the faithful method matches all four '
                               'source answers and each named method matches the complete answer but misses a part',
                          answers={m: {q: answer(m, q, i) for q in QUERIES} for m in ['source', args.faithful, *args.hidden]},
                          unedited_logit=float(values[('none', 'full')][i]),
                          logits={m: {q: float(values[(m, q)][i]) for q in QUERIES} for m in ['source', args.faithful, *args.hidden]},
                          text=r.get('text', '')[:600])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=1))
            print(json.dumps({k: v for k, v in record.items() if k != 'text'}, indent=1))
            print(record['text'][:400])
            return
    raise SystemExit('no row satisfies the rule')


if __name__ == '__main__':
    main()
