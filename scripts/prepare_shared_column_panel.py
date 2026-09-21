from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from prepare_source_column_reuse import save


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/final_science_20260921_round04'


def records(value):
    if isinstance(value, dict):
        if 'document_sha256' in value or ('text' in value and ('pair' in value or 'verb' in value)):
            yield value
        for item in value.values():
            yield from records(item)
    elif isinstance(value, list):
        for item in value:
            yield from records(item)


def main():
    c = json.loads((ROOT/'configs/fs04_human_source_columns_fit.json').read_text())
    panel_paths = {p.resolve() for p in (ROOT/'artifacts').rglob('*.json') if 'panel' in p.name.lower()}
    human = OUT/'SHARED_COLUMN_HUMAN_CONFIRMATION_PANEL.json'
    panel_paths.discard(human.resolve())
    for config in (ROOT/'configs').glob('*.json'):
        value = json.loads(config.read_text(encoding='utf-8-sig'))
        if not isinstance(value, dict):
            continue
        for key in ['evaluation_panel', 'panel', 'fit_panel']:
            path = value.get(key)
            if isinstance(path, str) and Path(path).is_file():
                panel_paths.add(Path(path).resolve())
        if isinstance(value.get('frozen_source_run'), str):
            path = Path(value['frozen_source_run'])/'panel.json'
            if path.is_file():
                panel_paths.add(path.resolve())
    excluded, verbs, nouns, texts = set(), set(), set(), set()
    for path in sorted(panel_paths):
        for row in records(json.loads(path.read_text(encoding='utf-8-sig'))):
            if 'document_sha256' in row:
                excluded.add(row['document_sha256'])
            if isinstance(row.get('text'), str):
                texts.add(row['text'])
                excluded.add(hashlib.sha256(row['text'].encode()).hexdigest())
            if 'verb' in row:
                verbs.add(row['verb'])
            if 'noun' in row:
                nouns.add(row['noun'])
            if isinstance(row.get('pair'), str) and ':' in row['pair']:
                verb, noun = row['pair'].split(':', 1)
                verbs.add(verb)
                nouns.add(noun)
    original = json.loads((ROOT/'configs/reform_r58_shift_source_development_v2.json').read_text())
    table = pq.read_table(original['dev_data'], filters=[('profession', 'in', [5, 25, 12, 24])],
                          columns=['hard_text', 'profession', 'gender'])
    buckets = {(p, g): [] for p in [5, 25, 12, 24] for g in [0, 1]}
    for row in table.to_pylist():
        text = row['hard_text']
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in excluded:
            continue
        excluded.add(digest)
        buckets[row['profession'], row['gender']].append(dict(text=text, document_sha256=digest,
                    profession=row['profession'], gender=row['gender'], split='dev'))
    selected = []
    for key, rows in buckets.items():
        assert len(rows) >= 16, (key, len(rows))
        selected += sorted(rows, key=lambda r: hashlib.sha256(('fs04shared/'+r['document_sha256']).encode()).hexdigest())[:16]
    tokenizer = AutoTokenizer.from_pretrained(c['model_local_dir'], local_files_only=True)
    tokens = tokenizer([r['text'] for r in selected], truncation=True, max_length=2048, add_special_tokens=True)['input_ids']
    for i, (row, token) in enumerate(zip(selected, tokens)):
        row.update(row_id=i, tokens=token)
    human_value = dict(rows=selected, original_split='dev', evidence='Fresh documents selected without model outcomes',
                       remaining_counts={str(k): len(v) for k, v in buckets.items()}, dataset_revision=c['dataset_revision'])
    if human.exists():
        assert json.loads(human.read_text()) == human_value
    new_verbs = ['commissioned', 'empowered', 'prompted', 'nudged', 'cautioned', 'trusted']
    new_nouns = ['biologists', 'architects', 'recruits', 'editors']
    assert not verbs.intersection(new_verbs), verbs.intersection(new_verbs)
    assert not nouns.intersection(new_nouns), nouns.intersection(new_nouns)
    grammar = []
    for verb in new_verbs:
        for noun in new_nouns:
            forms = [f'The {noun} had been {verb}', f'The committee {verb} the {noun}',
                     f'Yesterday the {noun} were {verb}', f'At the meeting, the supervisor {verb} the {noun}']
            for form, text in enumerate(forms):
                assert text not in texts
                grammar.append(dict(text=text, pair=f'{verb}:{noun}', verb=verb, noun=noun,
                                    form=form, role='predicate' if form % 2 == 0 else 'object'))
    rng = np.random.default_rng(2026092144)
    req = dict(queries=c['queries'][:7], dose_queries={}, member_queries={}, families={q: 'endpoints' for q in c['queries'][:7]})
    iq = dict(predicate=[1, 1, 1, 0], object=[0, 0, 0, 1], full=[1, 1, 1, 1])
    iff = {q: 'endpoints' for q in iq}
    groups = ['pronouns', 'names', 'associated_words']
    for kind in ['interior', 'boundary']:
        for i in range(6):
            q = f'{kind}_{i:02d}'
            v, u = rng.uniform(.05, .95, 3), rng.uniform(.05, .95, 2)
            if kind == 'boundary':
                v[i % 3], u[i % 2] = (i // 3) % 2, (i // 2) % 2
            req['queries'].append(q)
            req['dose_queries'][q] = dict(zip(groups, v.tolist()))
            req['families'][q] = kind
            iq[q], iff[q] = [float(u[0])]*3+[float(u[1])], kind
    source = json.loads(Path(c['source_manifest']).read_text())
    for i in range(8):
        q = f'member_subset_{i:02d}'
        req['queries'].append(q)
        req['families'][q] = 'member_subsets'
        req['member_queries'][q] = {s: rng.integers(0, 2, len(ids)).tolist() for s, ids in source['members'].items()}
    for bits in range(1, 16):
        v = [(bits >> i) & 1 for i in range(4)]
        if v in list(iq.values())[:3]:
            continue
        q = f'member_subset_{bits:02d}'
        iq[q], iff[q] = v, 'member_subsets'
    requests = OUT/'SHARED_COLUMN_HUMAN_CONFIRMATION_REQUESTS.json'
    if not human.exists():
        save(human, human_value)
    save(requests, req)
    infinitive = OUT/'SHARED_COLUMN_INFINITIVE_CONFIRMATION_PANEL.json'
    save(infinitive, dict(rows=grammar, queries=iq, families=iff, answer=' to'))
    save(OUT/'SHARED_COLUMN_PANEL_PROVENANCE.json', dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        human_contexts=len(selected), infinitive_contexts=len(grammar), new_verbs=new_verbs, new_nouns=new_nouns,
        exclusion_files=list(map(str, sorted(panel_paths))), prior_document_or_text_hashes=len(excluded),
        selection='Metadata and original dataset only; fixed hash order and balanced profession/gender cells; model outcomes unavailable.',
        scope='Human original-dev split with prior panels excluded; grammar new lexical items in four existing forms. Fresh continuous requests; grammar binary subsets exhaust the existing family.',
        files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [human, requests, infinitive]}))
    print(json.dumps(dict(human=len(selected), grammar=len(grammar), exclusions=len(panel_paths))))


if __name__ == '__main__':
    main()
