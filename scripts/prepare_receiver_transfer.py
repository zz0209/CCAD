from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile

import numpy as np
import torch
from transformers import AutoTokenizer


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    assert not path.exists(), str(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/scientific_reform_20260923/receiver_transfer'))
    parser.add_argument('--panel', type=Path, default=Path('artifacts/final_science_20260920_round02/AGREEMENT_DEVELOPMENT.json'))
    parser.add_argument('--archive', type=Path, default=Path('D:/CCAD_Storage/references/feature_circuits_shift/saes/50a434461d36ed78d1b0b901944e6edc829f1dce/dictionaries_pythia-70m-deduped_10.zip'))
    parser.add_argument('--model', type=Path, default=Path('D:/CCAD_Storage/models/pythia-70m-deduped/e93a9faa9c77e5d09219f6c868bfc7a1bd65593c'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('sources.npz', 'panel.json', 'smoke_panel.json', 'manifest.json'):
        assert not (args.output / name).exists(), name
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    that_ids = tokenizer.encode(' that', add_special_tokens=False)
    assert len(that_ids) == 1, that_ids
    original = json.loads(args.panel.read_text(encoding='utf-8'))
    rows = []
    for source_index, pair in enumerate(original['rows']):
        if pair['structure'] not in ('within_rc', 'rc'):
            continue
        pair_hash = digest(json.dumps(pair, sort_keys=True, ensure_ascii=False).encode())
        for direction, other in (('clean', 'patch'), ('patch', 'clean')):
            text, donor_text = pair[f'{direction}_prefix'], pair[f'{other}_prefix']
            ids = tokenizer.encode(text, add_special_tokens=False)
            donor_ids = tokenizer.encode(donor_text, add_special_tokens=False)
            positions = [i for i, token in enumerate(ids) if token == that_ids[0]]
            donor_positions = [i for i, token in enumerate(donor_ids) if token == that_ids[0]]
            assert len(positions) == len(donor_positions) == 1, (source_index, direction)
            answer, donor_answer = pair[f'{direction}_answer'], pair[f'{other}_answer']
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            donor_answer_ids = tokenizer.encode(donor_answer, add_special_tokens=False)
            assert len(answer_ids) == len(donor_answer_ids) == 1, (answer, donor_answer)
            rows.append(dict(row_id=len(rows), structure=pair['structure'], direction=direction,
                source_row_index=source_index, source_pair_sha256=pair_hash,
                pair_id=f"{pair['structure']}_{source_index}", original_pair=pair,
                text=text, donor_text=donor_text, input_ids=ids, donor_input_ids=donor_ids,
                tokens=tokenizer.convert_ids_to_tokens(ids), donor_tokens=tokenizer.convert_ids_to_tokens(donor_ids),
                a_position=positions[0], b_position=len(ids)-1,
                donor_a_position=donor_positions[0], donor_b_position=len(donor_ids)-1,
                answer=answer, donor_answer=donor_answer, answer_id=answer_ids[0], donor_answer_id=donor_answer_ids[0],
                document_sha256=digest(text.encode()), donor_document_sha256=digest(donor_text.encode())))
    assert len(rows) == 2 * sum(p['structure'] in ('within_rc', 'rc') for p in original['rows'])
    document_counts = Counter(row['document_sha256'] for row in rows)
    unordered_pairs = Counter(tuple(sorted((row['document_sha256'], row['donor_document_sha256']))) for row in rows)
    dependence = dict(row_count=len(rows), source_pair_count=len(rows)//2,
        unique_documents=len(document_counts), document_multiplicities=dict(document_counts),
        unordered_document_pairs=[dict(documents=list(key), directional_rows=count) for key, count in unordered_pairs.items()],
        structure_direction_counts={f'{structure}/{direction}': sum(r['structure']==structure and r['direction']==direction for r in rows)
            for structure in ('within_rc', 'rc') for direction in ('clean', 'patch')})
    common = dict(evidence_level='exposed_development', source_panel=str(args.panel.resolve()),
        source_panel_sha256=digest(args.panel.read_bytes()), model_local_dir=str(args.model),
        model_revision=args.model.name, tokenization='No added special tokens; original author prefixes and single-token answers',
        margin='recipient original answer logit minus the paired alternate answer logit',
        a_site='resid_3', a_members=[18529], b_site='attn_4', b_members=[3982, 31148],
        position_semantics='A is the unique that token in each prefix; B is its final token',
        selection='All within_rc and rc source rows, in saved order, with both clean and patch directions; no response-based selection')
    write_json(args.output/'panel.json', dict(common, rows=rows, dependence=dependence))
    smoke = []
    for structure in ('within_rc', 'rc'):
        pair_ids = list(dict.fromkeys(r['pair_id'] for r in rows if r['structure']==structure))[:2]
        smoke.extend(r for r in rows if r['pair_id'] in pair_ids)
    assert len(smoke) == 8
    write_json(args.output/'smoke_panel.json', dict(common, rows=smoke, parent_panel=str((args.output/'panel.json').resolve())))
    annotations_path = Path('artifacts/morning_reform_20260916/independent_circuit_reading/annotations/10_32768.jsonl')
    annotations = {item['Name']: item for item in map(json.loads, annotations_path.read_text().splitlines())}
    arrays, sources = {}, []
    with zipfile.ZipFile(args.archive) as archive:
        for site, member_ids, layer_path in (('resid_3', [18529], 'resid_out_layer3'), ('attn_4', [3982, 31148], 'attn_out_layer4')):
            member_path = f'dictionaries/pythia-70m-deduped/{layer_path}/10_32768/ae.pt'
            raw = archive.read(member_path)
            state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            selected = dict(ids=np.array(member_ids, dtype=np.int64),
                encoder=state['encoder.weight'][member_ids].numpy(),
                encoder_bias=state['encoder.bias'][member_ids].numpy(),
                decoder=state['decoder.weight'][:, member_ids].T.numpy(), center=state['bias'].numpy())
            for key, value in selected.items():
                arrays[f'{site}__{key}'] = value
            sources.append(dict(site=site, ids=member_ids, archive_member=member_path,
                archive_member_sha256=digest(raw), archive_member_bytes=len(raw),
                annotations={str(i): annotations[f'{site}/{i}'] for i in member_ids},
                array_shapes={key:list(value.shape) for key,value in selected.items()}))
            print(f'Extracted {site}, {len(member_ids)} source members', flush=True)
            del state, raw
    np.savez_compressed(args.output/'sources.npz', **arrays)
    license_path = Path('artifacts/morning_reform_20260916/independent_circuit_reading/LICENSE')
    manifest = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), generator=str(Path(__file__).resolve()),
        generator_sha256=digest(Path(__file__).read_bytes()), source_archive=str(args.archive),
        archive_revision=args.archive.parent.name, archive_bytes=args.archive.stat().st_size,
        source_code_commit='cf080789e16f50238097db1fb9a3947a6cfcf9f6',
        license='MIT, retained public feature-circuits source and archive assets',
        license_path=str(license_path.resolve()), license_sha256=digest(license_path.read_bytes()),
        source_panel=common['source_panel'], source_panel_sha256=common['source_panel_sha256'],
        model_local_dir=str(args.model), model_revision=args.model.name, sources=sources,
        outputs={name:dict(path=str((args.output/name).resolve()), sha256=digest((args.output/name).read_bytes()),
            bytes=(args.output/name).stat().st_size) for name in ('sources.npz', 'panel.json', 'smoke_panel.json')},
        dependence=dependence, smoke_rows=len(smoke), inference_performed=False)
    write_json(args.output/'manifest.json', manifest)
    print(json.dumps(dict(output=str(args.output), rows=len(rows), smoke_rows=len(smoke),
                         unique_documents=len(document_counts), counts=dependence['structure_direction_counts'])))


if __name__ == '__main__':
    main()
