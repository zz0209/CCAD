from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timezone
import argparse
import hashlib
import json
import string

from transformers import AutoTokenizer


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare(config):
    tokenizer = AutoTokenizer.from_pretrained(config['model_local_dir'], local_files_only=True)
    candidates = {}
    for token_id in sorted(tokenizer.get_vocab().values()):
        text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        family = text.strip().casefold()
        if not 3 <= len(family) <= 12 or any(char not in string.ascii_lowercase for char in family):
            continue
        normalized = ' ' + family
        ids = tokenizer.encode(normalized, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids, clean_up_tokenization_spaces=False) != normalized:
            continue
        candidates.setdefault(family, int(ids[0]))
    strata = defaultdict(list)
    for family, token_id in candidates.items():
        identity = hashlib.sha256(('CC23_functional_fragments|' + family).encode()).hexdigest()
        strata[family[0]].append(dict(word=family, token_id=token_id, word_family_sha256=identity))
    queues = {letter: deque(sorted(strata[letter], key=lambda row: (row['word_family_sha256'], row['token_id']))) for letter in string.ascii_lowercase}
    ordered = []
    while any(queues.values()):
        for letter in string.ascii_lowercase:
            if queues[letter]:
                ordered.append(queues[letter].popleft())
    counts = dict(mean=256, fit=2048, calibration=8, development=256, confirmation=512)
    assigned, offset = {}, 0
    for split, count in counts.items():
        assigned[split] = ordered[offset:offset+count]
        offset += count
    demos = sorted(assigned['fit'], key=lambda row: row['word_family_sha256'])[:10]
    assert len(demos) == 10
    demo_ids = {row['word_family_sha256'] for row in demos}
    prefix = ''.join(f'Word: {row["word"]}\nFirst letter: {row["word"][0].upper()}\nLast letter: {row["word"][-1].upper()}\n\n' for row in demos)
    answers = [tokenizer.encode(' ' + letter, add_special_tokens=False) for letter in string.ascii_uppercase]
    assert all(len(ids) == 1 for ids in answers)
    rows = []
    for split, allocated in assigned.items():
        for allocated_row in allocated:
            if allocated_row['word_family_sha256'] in demo_ids:
                continue
            row = dict(allocated_row, split=split, row_id=len(rows))
            common = prefix + 'Word: ' + row['word']
            common_ids = tokenizer.encode(common, add_special_tokens=False)
            word_start, word_end = len(common)-len(row['word']), len(common)
            prompts = {}
            for task in ('first', 'last'):
                prompt = common + ('\nFirst letter:' if task == 'first' else '\nLast letter:')
                encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
                positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > word_start and a < word_end]
                assert len(positions) == 1
                position = positions[0]
                assert encoded['input_ids'][position] == row['token_id']
                assert encoded['input_ids'][:len(common_ids)] == common_ids
                label = row['word'][0 if task == 'first' else -1].upper()
                label_id = answers[string.ascii_uppercase.index(label)][0]
                for letter, answer in zip(string.ascii_uppercase, answers):
                    assert tokenizer.encode(prompt + ' ' + letter, add_special_tokens=False) == encoded['input_ids'] + answer
                prompts[task] = dict(text=prompt, tokens=encoded['input_ids'], word_position=position, answer_id=label_id)
            assert prompts['first']['word_position'] == prompts['last']['word_position']
            row.update(first_label=string.ascii_lowercase.index(row['word'][0]), last_label=string.ascii_lowercase.index(row['word'][-1]), common_prefix=common, common_prefix_tokens=common_ids, prompts=prompts)
            rows.append(row)
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), scope='CCAD first/last-letter functional-fragment adaptation; tokenizer-derived words and explicit independent word-family splits', model_revision=config['model_revision'], model_local_dir=config['model_local_dir'], tokenizer_sha256=digest(Path(config['model_local_dir'])/'tokenizer.json'), code_sha256=digest(__file__), model_license='Apache-2.0', vocabulary_source='Pinned Qwen tokenizer; no external word list or SAEBench code copied', split_rule='SHA256 CC23_functional_fragments|strip+casefold; within-letter hash order; round-robin a-z; consecutive split allocations', requested_counts=counts, allocated_counts={key:len(value) for key,value in assigned.items()}, scored_counts={key:sum(row['split']==key for row in rows) for key in counts}, available_families=len(candidates), available_by_first_letter={key:len(strata[key]) for key in string.ascii_lowercase}, icl_families=demos, assignment={key:[row['word_family_sha256'] for row in value] for key,value in assigned.items()}, answer_ids=[ids[0] for ids in answers], rows=rows)
    destination = Path(config['output'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(path=str(destination), families=len(candidates), allocated=result['allocated_counts'], scored=result['scored_counts']), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
