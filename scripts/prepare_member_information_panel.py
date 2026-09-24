from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / 'artifacts/science_upgrade_20260919/ROUND04_INFINITIVE_PANEL.json'
VERBS = ['trained', 'motivated', 'recruited', 'hired', 'selected', 'paid']
NOUNS = ['paramedics', 'translators', 'plumbers', 'accountants']
PANEL_KEYS = ['panel', 'fit_panel', 'grammar_panel']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def identity(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def canonical_text(value):
    return ' '.join(value.casefold().split())


def lexical_identity(row):
    fields = row['pair'].split(':')
    assert len(fields) == 2 and all(fields), row
    verb, noun = fields
    assert row.get('verb', verb) == verb and row.get('noun', noun) == noun, row
    return verb.casefold(), noun.casefold()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/submission_round_20260924/member_information')
    args = parser.parse_args()
    output_names = ['confirmation_panel.json', 'development_reference.json', 'INPUT_PREPARATION.json']
    assert not any((args.output / name).exists() for name in output_names), args.output
    configurations = sorted((ROOT / 'configs').glob('*infinitive*.json'))
    assert configurations
    references = {}
    config_records = []
    for config_path in configurations:
        config = read_json(config_path)
        links = {}
        for key in PANEL_KEYS:
            if key not in config:
                continue
            panel_path = Path(config[key])
            if not panel_path.is_absolute():
                panel_path = ROOT / panel_path
            panel_path = panel_path.resolve()
            assert panel_path.is_file(), (config_path, key, panel_path)
            references.setdefault(panel_path, []).append({'config': str(config_path), 'key': key})
            links[key] = str(panel_path)
        config_records.append({'identity': identity(config_path), 'panel_fields': links})
    assert DEVELOPMENT.resolve() in references
    history_verbs, history_nouns, history_texts = set(), set(), set()
    history_records = []
    for panel_path, linked_by in sorted(references.items()):
        panel = read_json(panel_path)
        rows = panel['rows']
        assert rows, panel_path
        verbs, nouns, texts = set(), set(), set()
        for row in rows:
            verb, noun = lexical_identity(row)
            verbs.add(verb)
            nouns.add(noun)
            texts.add(canonical_text(row['text']))
        history_verbs.update(verbs)
        history_nouns.update(nouns)
        history_texts.update(texts)
        history_records.append({'identity': identity(panel_path), 'linked_by': linked_by,
                                'rows': len(rows), 'unique_texts': len(texts),
                                'verbs': sorted(verbs), 'nouns': sorted(nouns)})
    assert not set(VERBS) & history_verbs, sorted(set(VERBS) & history_verbs)
    assert not set(NOUNS) & history_nouns, sorted(set(NOUNS) & history_nouns)
    development = read_json(DEVELOPMENT)
    assert len(development['queries']) == 39
    assert list(development['queries']) == list(development['families'])
    templates = {}
    roles = {}
    for row in development['rows']:
        verb, noun = lexical_identity(row)
        template = row['text'].replace(verb, '{verb}').replace(noun, '{noun}')
        assert template.count('{verb}') == template.count('{noun}') == 1
        form = row['form']
        if form in templates:
            assert templates[form] == template and roles[form] == row['role']
        templates[form], roles[form] = template, row['role']
    assert sorted(templates) == [0, 1, 2, 3]
    rows = [{'pair': f'{verb}:{noun}', 'verb': verb, 'noun': noun, 'form': form,
             'role': roles[form], 'text': templates[form].format(verb=verb, noun=noun)}
            for verb in VERBS for noun in NOUNS for form in sorted(templates)]
    texts = {canonical_text(row['text']) for row in rows}
    assert len(rows) == len(texts) == 96
    assert not texts & history_texts
    written_at = datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
    confirmation = {'rows': rows, 'queries': development['queries'],
                    'families': development['families'], 'answer': development['answer'],
                    'scope': '六个历史未用动词与四个历史未用名词交叉，保留原四句式及39项请求；用于冻结成员信息比较。',
                    'prepared_at_utc': written_at}
    development_record = {'written_at_utc': written_at, 'identity': identity(DEVELOPMENT),
                          'role': '既有曝光输入，仅用于开发及运行核验。',
                          'rows': len(development['rows']), 'requests': len(development['queries'])}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in [('confirmation_panel.json', confirmation),
                          ('development_reference.json', development_record)]:
        with (args.output / name).open('x', encoding='utf-8') as stream:
            json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    record = {'written_at_utc': written_at, 'status': 'PASS', 'script': identity(Path(__file__)),
              'search_scope': 'configs目录中文件名含infinitive的全部JSON配置，以及它们直接指定的panel、fit_panel和grammar_panel；只读取输入JSON，不读取模型输出。',
              'configurations': config_records, 'historical_panels': history_records,
              'history': {'verbs': sorted(history_verbs), 'nouns': sorted(history_nouns),
                          'unique_normalized_texts': len(history_texts)},
              'selection': {'verbs': VERBS, 'nouns': NOUNS,
                            'rationale': '动词均可用于被动person-to-infinitive及主动verb-person-to-infinitive句式；名词均为复数职业人员。选择依据为语法合理性与已记录输入排除，未观察模型结果。',
                            'templates': templates, 'roles': roles},
              'checks': {'historical_verb_overlap': 0, 'historical_noun_overlap': 0,
                         'historical_text_overlap': 0, 'unique_contexts': 96,
                         'request_values_and_order_equal_to_development': True,
                         'family_values_and_order_equal_to_development': True,
                         'model_calls': 0},
              'outputs': {name: identity(args.output / name) for name in output_names[:2]}}
    with (args.output / 'INPUT_PREPARATION.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'PASS', 'configurations': len(configurations),
                      'historical_panels': len(references), 'history': record['history'],
                      'checks': record['checks'], 'confirmation': record['outputs']['confirmation_panel.json']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
