from pathlib import Path
import hashlib
import io
import json
import zipfile
import numpy as np
import torch


def main():
    root = Path('artifacts/final_science_20260920_round02')
    root.mkdir(exist_ok=True)
    assets = Path('artifacts/morning_reform_20260916/independent_circuit_reading')
    annotations = [json.loads(line) for line in (assets/'annotations/10_32768.jsonl').read_text().splitlines()]
    members = [8913, 13081, 11586, 14719, 16089, 26914, 5798, 15313, 22614, 31158]
    selected = {int(a['Name'].split('/')[1]):a for a in annotations
                if a['Name'].startswith('resid_4/') and int(a['Name'].split('/')[1]) in members}
    assert len(selected)==len(members)
    singular = [13081, 16089, 5798, 15313, 31158]
    plural = [i for i in members if i not in singular]
    queries = dict(singular=[float(i in singular) for i in members],
                   plural=[float(i in plural) for i in members], full=[1.]*len(members))
    rng = np.random.default_rng(922)
    for i, row in enumerate(rng.uniform(.1,.9,(8,len(members)))):
        queries[f'participation_{i:02d}']=row.tolist()
    for i in range(len(members)):
        queries[f'member_{members[i]}']=[float(j==i) for j in range(len(members))]
    fit = []
    development = []
    confirmation = []
    seen = set()
    for structure in ['simple','within_rc','rc']:
        rows = []
        path = assets/f'data/{structure}_train.json'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records.sort(key=lambda r: hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest())
        for item in records:
            digest = hashlib.sha256(item['clean_prefix'].encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            rows.append(dict(item, text=item['clean_prefix'], structure=structure, document_sha256=digest))
        rows.sort(key=lambda r:r['document_sha256'])
        counts = (8,8,24) if structure=='simple' else (20,20,64)
        nfit, ndev, nconfirm = counts
        assert len(rows)>=sum(counts)
        fit.extend(rows[:nfit])
        development.extend(rows[nfit:nfit+ndev])
        confirmation.extend(rows[nfit+ndev:sum(counts)])
    for label, rows in [('fit',fit),('development',development),('confirmation',confirmation)]:
        path=root/f'AGREEMENT_{label.upper()}.json'
        if path.exists(): raise FileExistsError(path)
        path.write_text(json.dumps(dict(rows=rows,queries=queries,members=members,
            groups=dict(singular=singular,plural=plural),
            scope='Published resid4 subject-number annotations, zero-deletion partial programs on original clean prefixes. Target transfer evaluation; not the full SFC circuit benchmark.'),indent=2)+'\n')
    c=json.loads(Path('configs/functional_pursuit_smoke_20260920.json').read_text())
    with zipfile.ZipFile(c['source_archive']) as archive:
        data=archive.read('dictionaries/pythia-70m-deduped/resid_out_layer4/10_32768/ae.pt')
    sd=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
    path=root/'agreement_source_parts.npz'
    if path.exists(): raise FileExistsError(path)
    np.savez(path,encoder=sd['encoder.weight'][members].numpy(),encoder_bias=sd['encoder.bias'][members].numpy(),
             decoder=sd['decoder.weight'][:,members].T.numpy(),center=sd['bias'].numpy())
    manifest=dict(members=members,annotations=selected,source_archive=c['source_archive'],
                  archive_member_sha256=hashlib.sha256(data).hexdigest(),
                  license='MIT official source repository and retained assets',
                  fit=len(fit),development=len(development),confirmation=len(confirmation),seed=922,
                  sampling='One answer pair per unique clean prefix, chosen by record SHA256 before prefix-hash splits',
                  source_function='Singular/plural subject-number parts at resid4, ten published annotations',
                  target_responses_seen=False)
    (root/'AGREEMENT_SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='annotations'},indent=2))


if __name__=='__main__':
    main()
