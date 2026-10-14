import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

DESCRIPTION = ('Build the P2 two-hop task: 6 relation pairs over countries (hop 1 identifies a country by its capital, '
               'official language or currency; hop 2 asks its language, currency, capital or continent), 25 prompts '
               'per pair with a one-hop control, bridge countries that are single tokens under every tokenizer, one '
               'swap target per prompt with a different answer, and 2 relation pairs held out from concept-vector '
               'fitting. Prompts carry a fixed two-shot prefix built from countries that never act as bridges.')

# name, capital answers (None = capital repeats the country name or is disputed), official-language answers (None =
# several official languages), language that identifies the country (None = shared), currency answers, currency
# that identifies the country (None = shared), continent answers (None = transcontinental).
# Ordered roughly by prominence; each pair takes the first 25 eligible countries.
COUNTRIES = [
    ('France', ['Paris'], ['French'], None, ['euro'], None, ['Europe']),
    ('Germany', ['Berlin'], ['German'], None, ['euro'], None, ['Europe']),
    ('Japan', ['Tokyo'], ['Japanese'], 'Japanese', ['yen'], 'yen', ['Asia']),
    ('China', ['Beijing'], ['Chinese', 'Mandarin'], 'Chinese', ['yuan', 'renminbi'], 'yuan', ['Asia']),
    ('Italy', ['Rome'], ['Italian'], None, ['euro'], None, ['Europe']),
    ('Spain', ['Madrid'], ['Spanish'], None, ['euro'], None, ['Europe']),
    ('Russia', ['Moscow'], ['Russian'], 'Russian', ['ruble', 'rouble'], 'ruble', None),
    ('India', ['New Delhi', 'Delhi'], None, 'Hindi', ['rupee'], None, ['Asia']),
    ('Brazil', ['Brasília', 'Brasilia'], ['Portuguese'], None, ['real'], 'real', ['South America']),
    ('Mexico', None, ['Spanish'], None, ['peso'], None, ['North America']),
    ('Canada', ['Ottawa'], None, None, ['dollar'], None, ['North America']),
    ('Egypt', ['Cairo'], ['Arabic'], None, ['pound'], None, ['Africa']),
    ('Greece', ['Athens'], ['Greek'], 'Greek', ['euro'], None, ['Europe']),
    ('Turkey', ['Ankara'], ['Turkish'], 'Turkish', ['lira'], 'lira', None),
    ('Poland', ['Warsaw'], ['Polish'], 'Polish', ['zloty', 'złoty'], 'zloty', ['Europe']),
    ('Sweden', ['Stockholm'], ['Swedish'], 'Swedish', ['krona'], 'krona', ['Europe']),
    ('Norway', ['Oslo'], ['Norwegian'], 'Norwegian', ['krone'], None, ['Europe']),
    ('Denmark', ['Copenhagen'], ['Danish'], 'Danish', ['krone'], None, ['Europe']),
    ('Finland', ['Helsinki'], None, 'Finnish', ['euro'], None, ['Europe']),
    ('Netherlands', ['Amsterdam'], ['Dutch'], None, ['euro'], None, ['Europe']),
    ('Portugal', ['Lisbon'], ['Portuguese'], None, ['euro'], None, ['Europe']),
    ('Switzerland', ['Bern', 'Berne'], None, None, ['franc'], None, ['Europe']),
    ('Belgium', ['Brussels'], None, None, ['euro'], None, ['Europe']),
    ('Hungary', ['Budapest'], ['Hungarian'], 'Hungarian', ['forint'], 'forint', ['Europe']),
    ('Ukraine', ['Kyiv', 'Kiev'], ['Ukrainian'], 'Ukrainian', ['hryvnia'], 'hryvnia', ['Europe']),
    ('Ireland', ['Dublin'], None, None, ['euro'], None, ['Europe']),
    ('Austria', ['Vienna'], ['German'], None, ['euro'], None, ['Europe']),
    ('Thailand', ['Bangkok'], ['Thai'], 'Thai', ['baht'], 'baht', ['Asia']),
    ('Vietnam', ['Hanoi'], ['Vietnamese'], 'Vietnamese', ['dong'], 'dong', ['Asia']),
    ('Indonesia', ['Jakarta'], ['Indonesian'], 'Indonesian', ['rupiah'], 'rupiah', ['Asia']),
    ('Iran', ['Tehran'], ['Persian', 'Farsi'], 'Persian', ['rial'], None, ['Asia']),
    ('Israel', ['Jerusalem'], ['Hebrew'], 'Hebrew', ['shekel'], 'shekel', ['Asia']),
    ('Pakistan', ['Islamabad'], None, 'Urdu', ['rupee'], None, ['Asia']),
    ('Iraq', ['Baghdad'], ['Arabic'], None, ['dinar'], None, ['Asia']),
    ('Argentina', ['Buenos Aires'], ['Spanish'], None, ['peso'], None, ['South America']),
    ('Peru', ['Lima'], ['Spanish'], None, ['sol'], 'sol', ['South America']),
    ('Chile', ['Santiago'], ['Spanish'], None, ['peso'], None, ['South America']),
    ('Colombia', ['Bogotá', 'Bogota'], ['Spanish'], None, ['peso'], None, ['South America']),
    ('Venezuela', ['Caracas'], ['Spanish'], None, ['bolívar', 'bolivar'], 'bolívar', ['South America']),
    ('Cuba', ['Havana'], ['Spanish'], None, ['peso'], None, None),
    ('Kenya', ['Nairobi'], None, None, ['shilling'], None, ['Africa']),
    ('Nigeria', ['Abuja'], ['English'], None, ['naira'], 'naira', ['Africa']),
    ('Ethiopia', ['Addis Ababa'], ['Amharic'], 'Amharic', ['birr'], 'birr', ['Africa']),
    ('Morocco', ['Rabat'], ['Arabic'], None, ['dirham'], None, ['Africa']),
    ('Ghana', ['Accra'], ['English'], None, ['cedi'], 'cedi', ['Africa']),
    ('Romania', ['Bucharest'], ['Romanian'], None, ['leu'], None, ['Europe']),
    ('Bulgaria', ['Sofia'], ['Bulgarian'], 'Bulgarian', ['lev'], 'lev', ['Europe']),
    ('Serbia', ['Belgrade'], ['Serbian'], 'Serbian', ['dinar'], None, ['Europe']),
    ('Croatia', ['Zagreb'], ['Croatian'], 'Croatian', None, None, ['Europe']),
    ('Iceland', ['Reykjavik', 'Reykjavík'], ['Icelandic'], 'Icelandic', ['krona', 'króna'], None, ['Europe']),
    ('Belarus', ['Minsk'], None, 'Belarusian', ['ruble', 'rouble'], None, ['Europe']),
    ('Bangladesh', ['Dhaka'], ['Bengali', 'Bangla'], 'Bengali', ['taka'], 'taka', ['Asia']),
    ('Nepal', ['Kathmandu'], ['Nepali'], 'Nepali', ['rupee'], None, ['Asia']),
    ('Philippines', ['Manila'], None, 'Filipino', ['peso'], None, ['Asia']),
    ('Malaysia', ['Kuala Lumpur'], ['Malay'], None, ['ringgit'], 'ringgit', ['Asia']),
    ('Afghanistan', ['Kabul'], None, 'Pashto', ['afghani'], 'afghani', ['Asia']),
    ('Cambodia', ['Phnom Penh'], ['Khmer'], 'Khmer', ['riel'], 'riel', ['Asia']),
    ('Myanmar', ['Naypyidaw', 'Naypyitaw'], ['Burmese'], 'Burmese', ['kyat'], 'kyat', ['Asia']),
    ('Armenia', ['Yerevan'], ['Armenian'], 'Armenian', ['dram'], 'dram', None),
    ('Georgia', ['Tbilisi'], ['Georgian'], 'Georgian', ['lari'], 'lari', None),
    ('Jordan', ['Amman'], ['Arabic'], None, ['dinar'], None, ['Asia']),
    ('Lebanon', ['Beirut'], ['Arabic'], None, ['pound'], None, ['Asia']),
    ('Syria', ['Damascus'], ['Arabic'], None, ['pound'], None, ['Asia']),
    ('Qatar', ['Doha'], ['Arabic'], None, ['riyal'], None, ['Asia']),
    ('Libya', ['Tripoli'], ['Arabic'], None, ['dinar'], None, ['Africa']),
    ('Somalia', ['Mogadishu'], None, 'Somali', ['shilling'], None, ['Africa']),
    ('Uganda', ['Kampala'], None, None, ['shilling'], None, ['Africa']),
    ('Sudan', ['Khartoum'], None, None, ['pound'], None, ['Africa']),
    ('Ecuador', ['Quito'], ['Spanish'], None, ['dollar'], None, ['South America']),
    ('Bolivia', ['Sucre', 'La Paz'], None, None, ['boliviano'], 'boliviano', ['South America']),
    ('Haiti', ['Port-au-Prince'], None, None, ['gourde'], 'gourde', ['North America']),
]
# Demonstration countries: multi-token names, so they can never be bridges or swap targets.
DEMOS = [('the Czech Republic', 'Prague', 'Czech', 'koruna', 'Europe'),
         ('South Korea', 'Seoul', 'Korean', 'won', 'Asia')]
ANSWER = dict(capital=1, language=2, currency=4, continent=6)
# Two-hop template, one-hop control template, hop-1 relation, hop-2 relation.
PAIRS = {
    'capital_to_language': ('The official language of the country whose capital is {h1} is',
                            'The official language of {country} is', 'capital', 'language'),
    'capital_to_currency': ('The currency of the country whose capital is {h1} is the',
                            'The currency of {country} is the', 'capital', 'currency'),
    'capital_to_continent': ('The country whose capital is {h1} is located on the continent of',
                             '{country} is located on the continent of', 'capital', 'continent'),
    'language_to_capital': ('The capital of the country whose official language is {h1} is',
                            'The capital of {country} is', 'language', 'capital'),
    'language_to_continent': ('The country whose official language is {h1} is located on the continent of',
                              '{country} is located on the continent of', 'language', 'continent'),
    'currency_to_capital': ('The capital of the country whose currency is the {h1} is',
                            'The capital of {country} is', 'currency', 'capital'),
}
# Confirmation set v2: one new wording per pair (same hop-1 and hop-2 relations), new demonstrations, seed 1.
PAIRS_V2 = {
    'capital_to_language': ('The language officially spoken in the country with the capital {h1} is',
                            'The language officially spoken in {country} is', 'capital', 'language'),
    'capital_to_currency': ('The money used in the country with the capital {h1} is called the',
                            'The money used in {country} is called the', 'capital', 'currency'),
    'capital_to_continent': ('The country with the capital {h1} lies on the continent of',
                             '{country} lies on the continent of', 'capital', 'continent'),
    'language_to_capital': ('The capital city of the country where the official language is {h1} is',
                            'The capital city of {country} is', 'language', 'capital'),
    'language_to_continent': ('The country where the official language is {h1} lies on the continent of',
                              '{country} lies on the continent of', 'language', 'continent'),
    'currency_to_capital': ('The capital city of the country that uses the {h1} as its currency is',
                            'The capital city of {country} is', 'currency', 'capital'),
}
DEMOS_V2 = [('North Macedonia', 'Skopje', 'Macedonian', 'denar', 'Europe'),
            ('Kyrgyzstan', 'Bishkek', 'Kyrgyz', 'som', 'Asia')]
# Concept-fitting set: a third wording of the four training pairs, never scored; v1 demonstrations.
PAIRS_FIT = {
    'capital_to_language': ('In the country whose capital city is {h1}, the official language is',
                            None, 'capital', 'language'),
    'capital_to_currency': ('In the country whose capital city is {h1}, people pay with the',
                            None, 'capital', 'currency'),
    'capital_to_continent': ('The nation with {h1} as its capital is part of the continent of',
                             None, 'capital', 'continent'),
    'language_to_capital': ('The nation whose official language is {h1} has its capital in',
                            None, 'language', 'capital'),
}
VERSIONS = dict(v1=(PAIRS, DEMOS, 0), v2=(PAIRS_V2, DEMOS_V2, 1))
HELD_OUT = ['language_to_continent', 'currency_to_capital']
PER_PAIR = 25


def hop1_text(c, rel):
    if rel == 'capital':
        return c[1][0] if c[1] else None
    return c[3] if rel == 'language' else c[5]


def fill(template, country):
    text = template.format(country=country)
    return text[0].upper() + text[1:]


def demo_prefix(two_hop, one_hop, h1, h2, demos=DEMOS):
    col = dict(capital=1, language=2, currency=3, continent=4)
    lines = []
    for d in demos:
        text = two_hop.format(h1=d[col[h1]]) if one_hop is None else fill(one_hop, d[0])
        lines.append(f'Fact: {text} {d[col[h2]]}.\n')
    return ''.join(lines)


def first_tokens(tok, answers):
    return {tok.encode(' ' + a, add_special_tokens=False)[0] for a in answers}


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--tokenizers', nargs='+', required=True, help='name=repo_or_path')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--version', choices=sorted(VERSIONS), default='v1')
    parser.add_argument('--concept_fit', action='store_true',
                        help='write the concept-fitting set (PAIRS_FIT over every pool country) instead of a task')
    parser.add_argument('--reference', type=Path, default=None, help='v1 task.json whose bridge pools must match')
    args = parser.parse_args()
    toks = {spec.split('=')[0]: AutoTokenizer.from_pretrained(spec.split('=')[1]) for spec in args.tokenizers}
    single = {c[0]: all(len(t.encode(' ' + c[0], add_special_tokens=False)) == 1 for t in toks.values())
              for c in COUNTRIES}
    print('multi-token country names dropped:', [n for n, ok in single.items() if not ok])
    pairs_spec, demos, seed = VERSIONS[args.version]
    for d in demos:
        name = d[0][4:] if d[0].startswith('the ') else d[0]
        if any(len(t.encode(' ' + name, add_special_tokens=False)) < 2 for t in toks.values()):
            raise ValueError(f'demonstration country {d[0]} is a single token')
    rng = random.Random(seed)

    def pool_of(h1, h2):
        def eligible(c):
            if not single[c[0]] or hop1_text(c, h1) is None or c[ANSWER[h2]] is None:
                return False
            # An answer that starts with the bridge's own token would make the swap ill-posed.
            return all(t.encode(' ' + c[0], add_special_tokens=False)[0] not in first_tokens(t, c[ANSWER[h2]])
                       for t in toks.values())
        return [c for c in COUNTRIES if eligible(c)]

    if args.concept_fit:
        countries = sorted({c[0] for _, _, h1, h2 in PAIRS.values() for c in pool_of(h1, h2)[:PER_PAIR]})
        items = []
        for pair, (two_hop, _, h1, h2) in PAIRS_FIT.items():
            prefix = demo_prefix(two_hop, None, h1, h2)
            for c in pool_of(h1, h2):
                if c[0] in countries:
                    items.append(dict(id=f'fit/{pair}/{c[0]}', pair=pair, country=c[0],
                                      prompt=prefix + 'Fact: ' + two_hop.format(h1=hop1_text(c, h1))))
        missing = sorted(set(countries) - {it['country'] for it in items})
        if missing:
            raise ValueError(f'pool countries without a concept-fitting prompt: {missing}')
        out = dict(kind='concept_fit', pairs={p: v[0] for p, v in PAIRS_FIT.items()}, demos=DEMOS,
                   countries=countries, items=items, tokenizers=dict(s.split('=') for s in args.tokenizers))
        args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(len(items), 'concept-fitting prompts over', len(countries), 'countries;', items[0]['prompt'])
        return
    items, pairs_out = [], {}
    for pair, (two_hop, one_hop, h1, h2) in pairs_spec.items():
        pool = pool_of(h1, h2)[:PER_PAIR]
        if len(pool) < PER_PAIR:
            raise ValueError(f'{pair}: only {len(pool)} eligible countries')
        pairs_out[pair] = dict(two_hop=two_hop, one_hop=one_hop, hop1=h1, hop2=h2, held_out=pair in HELD_OUT,
                               bridges=[c[0] for c in pool])
        prefix2 = demo_prefix(two_hop, None, h1, h2, demos)
        prefix1 = demo_prefix(two_hop, one_hop, h1, h2, demos)
        for c in pool:
            targets = [o for o in pool if o[0] != c[0] and all(
                not (first_tokens(t, c[ANSWER[h2]]) & first_tokens(t, o[ANSWER[h2]])) for t in toks.values())]
            if not targets:
                raise ValueError(f'{pair}/{c[0]}: no swap target with a distinct answer')
            o = rng.choice(targets)
            items.append(dict(
                id=f'{pair}/{c[0]}', pair=pair, held_out=pair in HELD_OUT, bridge=c[0], target=o[0],
                two_hop=prefix2 + 'Fact: ' + two_hop.format(h1=hop1_text(c, h1)),
                one_hop=prefix1 + 'Fact: ' + fill(one_hop, c[0]),
                target_one_hop=prefix1 + 'Fact: ' + fill(one_hop, o[0]),
                answers=c[ANSWER[h2]], target_answers=o[ANSWER[h2]]))
    for name, t in toks.items():
        n_tok = [len(t.encode(it['two_hop'])) for it in items]
        print(name, 'two-hop prompt tokens min/max', min(n_tok), max(n_tok))
    if args.reference is not None:
        ref = json.loads(args.reference.read_text())['pairs']
        if {p: v['bridges'] for p, v in ref.items()} != {p: v['bridges'] for p, v in pairs_out.items()}:
            raise ValueError('bridge pools differ from the reference task')
    out = dict(version=args.version, seed=seed, per_pair=PER_PAIR, held_out=HELD_OUT, demos=demos, pairs=pairs_out,
               items=items, tokenizers=dict(s.split('=') for s in args.tokenizers))
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(len(items), 'items;', {p: v['bridges'] for p, v in pairs_out.items()})
    for it in items[::25]:
        print('---', it['id'], it['answers'], '->', it['target'], it['target_answers'])
        print(it['two_hop'])
        print(it['one_hop'])


if __name__ == '__main__':
    main()
