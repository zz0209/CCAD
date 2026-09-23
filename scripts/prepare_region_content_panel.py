from pathlib import Path
import hashlib
import itertools
import json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/scientific_reform_20260923/content_transfer/PANEL.json'
VOCAB = {
    'development': [
        ('man', 'woman', 'men', 'women'), ('boy', 'girl', 'boys', 'girls'),
        ('father', 'mother', 'fathers', 'mothers'), ('son', 'daughter', 'sons', 'daughters')],
    'confirmation': [
        ('brother', 'sister', 'brothers', 'sisters'), ('uncle', 'aunt', 'uncles', 'aunts'),
        ('husband', 'wife', 'husbands', 'wives'), ('king', 'queen', 'kings', 'queens'),
        ('prince', 'princess', 'princes', 'princesses'), ('nephew', 'niece', 'nephews', 'nieces'),
        ('grandfather', 'grandmother', 'grandfathers', 'grandmothers'),
        ('grandson', 'granddaughter', 'grandsons', 'granddaughters')],
}
VERBS = {
    'development': (['greeted', 'visited'], ['said', 'claimed'],
                    ['injured', 'hurt', 'embarrassed', 'amused']),
    'confirmation': (['thanked', 'called', 'noticed', 'helped'], ['thought', 'believed'],
                     ['defended', 'blamed', 'introduced', 'praised', 'washed', 'dressed',
                      'photographed', 'criticized']),
}
LABELS = [' himself', ' herself', ' themselves']


def main():
    assert not OUTPUT.exists(), OUTPUT
    rows = []
    units = []
    for split, nouns in VOCAB.items():
        relatives, complements, finals = VERBS[split]
        offsets = [1, 3]
        for unit, (first, offset) in enumerate(itertools.product(range(len(nouns)), offsets)):
            second = (first + offset) % len(nouns)
            assert first != second
            lexical_id = f'{split}_{unit:02d}'
            identities = [nouns[first], nouns[second]]
            fixed_gender = [unit % 2, (unit // 2) % 2]
            units.append(dict(lexical_id=lexical_id, split=split, noun_families=[first, second],
                              nouns=identities, number_fixed_gender=fixed_gender))
            for syntax, factor in itertools.product(['relative', 'complement'], ['number', 'gender']):
                group = {}
                controller = 0 if syntax == 'relative' else 1
                for state in itertools.product(range(2), repeat=2):
                    words = [identities[k][fixed_gender[k] + 2 * state[k]] if factor == 'number'
                             else identities[k][state[k]] for k in range(2)]
                    final = finals[unit % len(finals)]
                    if syntax == 'relative':
                        text = f'The {words[0]} that the {words[1]} {relatives[unit % len(relatives)]} {final}'
                    else:
                        text = f'The {words[0]} {complements[unit % len(complements)]} that the {words[1]} {final}'
                    labels = [LABELS[fixed_gender[controller]], LABELS[2]] if factor == 'number' else LABELS[:2]
                    index = len(rows)
                    group[state] = index
                    rows.append(dict(id=index, split=split, lexical_id=lexical_id, syntax=syntax,
                                     factor=factor, states=list(state), controller=controller, text=text,
                                     answer=labels[state[controller]], label0=labels[0], label1=labels[1],
                                     noun_families=[first, second],
                                     text_sha256=hashlib.sha256(text.encode()).hexdigest()))
                for state, index in group.items():
                    rows[index]['donor_first'] = group[1-state[0], state[1]]
                    rows[index]['donor_second'] = group[state[0], 1-state[1]]
    for row in rows:
        for axis, name in enumerate(['donor_first', 'donor_second']):
            donor = rows[row[name]]
            assert donor['lexical_id'] == row['lexical_id']
            assert donor['syntax'] == row['syntax'] and donor['factor'] == row['factor']
            assert [a != b for a, b in zip(row['states'], donor['states'])] == [k == axis for k in range(2)]
            assert (donor['answer'] != row['answer']) == (axis == row['controller'])
    dev = {word for entry in VOCAB['development'] for word in entry}
    confirm = {word for entry in VOCAB['confirmation'] for word in entry}
    assert dev.isdisjoint(confirm)
    split_text = {s: {r['text_sha256'] for r in rows if r['split'] == s} for s in VOCAB}
    assert split_text['development'].isdisjoint(split_text['confirmation'])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(dict(rows=rows, lexical_units=units, labels=LABELS,
        scope='Controlled factorial examples inspired by reflexive agreement contrasts; not an official benchmark. All inputs retained independently of model predictions. Lexical units share noun families within split; uncertainty must retain this dependence.',
        token_boundary='R29 leading EOS; no trailing EOS before reflexive continuation',
        counts={s: sum(r['split'] == s for r in rows) for s in VOCAB}), indent=2), encoding='utf-8')
    print(json.dumps(dict(path=str(OUTPUT), rows=len(rows), sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
