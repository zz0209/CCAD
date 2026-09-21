import argparse
import hashlib
import importlib
import json
import pickle
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    output = Path(c['evaluation_panel'])
    assert not output.exists(), output
    root = Path(c['generator_adapter']).resolve()
    sys.path.insert(0, str(root))
    from utils.string_utils import string_beautify
    tokenizer = AutoTokenizer.from_pretrained(c['model_local_dir'], local_files_only=True)
    random.seed(c['generation_seed'])
    np.random.seed(c['generation_seed'])
    seen, files = set(), {}
    for manifest_path in c['exclusion_manifests']:
        for record in json.loads(Path(manifest_path).read_text())['files']:
            path = Path(record['path']).resolve()
            if str(path) in files:
                continue
            assert digest(path) == record['sha256'], path
            files[str(path)] = record['sha256']
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    seen.update([row['sentence_good'], row['sentence_bad']])
    for record in c.get('exclusion_panels', []):
        path = Path(record['path']).resolve()
        assert digest(path) == record['sha256'], path
        files[str(path)] = record['sha256']
        for row in json.loads(path.read_text())['rows']:
            seen.update([row['sentence_good'], row['sentence_bad']])
    rows, counts = [], {}
    checkpoint = output.parent/'generation_checkpoint.pkl'
    output.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        with checkpoint.open('rb') as stream:
            saved = pickle.load(stream)
        if saved['config_sha256'] != digest(args.config):
            previous_config = Path(c['generation_resume_config'])
            assert saved['config_sha256'] == digest(previous_config)
            previous = json.loads(previous_config.read_text())
            changed = {key for key in set(c) | set(previous) if c.get(key) != previous.get(key)}
            assert changed <= {'evaluation_panel', 'generator_adapter', 'generation_resume_config'}, changed
        rows, counts, seen = saved['rows'], saved['counts'], saved['seen']
        random.setstate(saved['random_state'])
        np.random.set_state(saved['numpy_state'])
    for task in c['tasks']:
        if counts.get(task, {}).get('accepted', 0) == c['eval_pairs_per_task']:
            continue
        module = importlib.import_module('generation_projects.blimp.'+task)
        name = 'AgreementGenerator' if task == 'regular_plural_subject_verb_agreement_1' else 'AnaphorGenerator'
        generator = getattr(module, name)()
        rejected = Counter(counts.get(task, {}).get('rejected', {}))
        accepted = counts.get(task, {}).get('accepted', 0)
        for attempt in range(c['eval_pairs_per_task']*100):
            data, reason = generator.sample()
            if data is None:
                assert reason in ['empty_noun_candidates', 'empty_reflexive_candidates', 'empty_mismatch_verb_candidates']
                rejected[reason] += 1
                continue
            for field in generator.data_fields:
                if field in data:
                    data[field] = string_beautify(data[field])
            good, bad = data['sentence_good'], data['sentence_bad']
            assert max(len(good), len(bad)) < 1024
            if good in seen or bad in seen:
                rejected['previous_or_duplicate_sentence'] += 1
                continue
            if good == bad:
                rejected['identical_pair'] += 1
                continue
            prefix = data['one_prefix_prefix']
            assert good.startswith(prefix) and bad.startswith(prefix)
            encoded = {key: [tokenizer.eos_token_id]+tokenizer.encode(data['sentence_'+key], add_special_tokens=False)+[tokenizer.eos_token_id] for key in ['good', 'bad']}
            prefix_tokens = [tokenizer.eos_token_id]+tokenizer.encode(prefix, add_special_tokens=False)
            assert encoded['good'][:len(prefix_tokens)] == prefix_tokens == encoded['bad'][:len(prefix_tokens)]
            seen.update([good, bad])
            rows.append(dict(task=task, row_id=accepted, split=c['evaluation_split'],
                sentence_good=good, sentence_bad=bad, good=encoded['good'], bad=encoded['bad'],
                position=len(prefix_tokens)-1, generator_metadata=generator.make_metadata_dict()))
            accepted += 1
            if accepted % 16 == 0:
                counts[task] = dict(accepted=accepted, rejected=dict(rejected))
                with checkpoint.open('wb') as stream:
                    pickle.dump(dict(config_sha256=digest(args.config), rows=rows, counts=counts,
                        seen=seen, random_state=random.getstate(), numpy_state=np.random.get_state()), stream)
                print(json.dumps(dict(task=task, accepted=accepted, requested=c['eval_pairs_per_task'])), flush=True)
            if accepted == c['eval_pairs_per_task']:
                break
        assert accepted == c['eval_pairs_per_task'], (task, accepted)
        counts[task] = dict(accepted=accepted, rejected=dict(rejected))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(rows=rows), indent=2)+'\n')
    manifest = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=digest(args.config), generator_commit='7b93dccc9a773d76cfdbc5391f7087e34eefd3cf',
        generation_seed=c['generation_seed'], excluded_files=files, counts=counts,
        output=dict(path=output.as_posix(), sha256=digest(output)),
        adapter_files={str(path.relative_to(root)): digest(path) for path in root.rglob('*.py')},
        semantics='Original accepted-pair sampling rules. Platform path, bounded Unicode storage, import guards and empty-exclusion set difference adapted manually. An empty noun candidate set returns a recorded rejected draw, matching the original outer generator rejection. No grammatical label or outcome changes.',
        scope='New draws from three original BLiMP grammars and vocabulary, with complete-sentence good-minus-bad log probabilities. Original benchmark and all supplied historical generated sentences excluded. No outcome-based selection.',
        license='BLiMP dataset CC BY 4.0. Upstream generator has no formal code license; adapter retained locally for internal research and excluded from code redistribution.')
    output.with_name('DATA_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
