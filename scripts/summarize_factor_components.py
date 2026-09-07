"""Descriptive block-aware source-component tables from persisted raw rows."""
import argparse,csv,json,statistics
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();groups={}
    for line in (args.run/'metrics.raw.jsonl').read_text().splitlines():
        row=json.loads(line);key=tuple(row.get(k) for k in ['checkpoint','factor','method','split']);groups.setdefault(key,[]).append(row)
    out=[]
    for key,rows in groups.items():
        row=dict(zip(['checkpoint','factor','method','split'],key),n=len(rows),blocks=len({r['block'] for r in rows}),correct=sum(r['donor_label_correct'] for r in rows))
        for metric in ['relative_vector_sqerror','delta_norm','raw_delta_norm','kl_to_reference','margin_change']:
            values=[r[metric] for r in rows if r[metric] is not None];row[metric]=statistics.mean(values) if values else None
        out.append(row)
    (args.run/'descriptive_table.json').write_text(json.dumps(dict(rows=out,scope='Descriptive totals; reciprocal directions and shared lexical blocks are not independent repeats.'),indent=2)+'\n')
    with (args.run/'descriptive_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
    for row in out:
        if row['factor']=='subject' and row['split']=='held_lexical_development':print(json.dumps(row))

if __name__=='__main__':main()
