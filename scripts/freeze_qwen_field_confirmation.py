import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def identity(path):
    path = Path(path)
    return dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    out = args.directory
    assert not (out/'CONFIRMATION_FREEZE.json').exists()
    triple_path = out/'UNSEEN_TRIPLE_PANEL.json'
    triple = json.loads(triple_path.read_text())
    old_path = Path('runs/REFORM_R38_qwen_member_fields_five_v1_20260914/panel.json')
    old = json.loads(old_path.read_text())
    rows, pairs = [], []
    templates = ['57+24=81\n16+35=51\n{a}+{b}=',
                 'Q: Add 57 and 24. A: 81\nQ: Add 16 and 35. A: 51\nQ: Add {a} and {b}. A:']
    selected = [p for p in old['pairs'] if p['template'] == 0][24:48]
    assert len(selected) == 24
    for template_id, template in enumerate(templates):
        for cluster, p in enumerate(selected):
            ids = []
            for key in ['recipient', 'donor']:
                row = copy.deepcopy(old['rows'][p[key]])
                row.update(template=template_id, split='confirmation', prompt=template.format(**row), arity=2)
                ids.append(len(rows))
                rows.append(row)
            pair = copy.deepcopy(p)
            pair.update(recipient=ids[0], donor=ids[1], template=template_id, arity=2, cluster=cluster)
            pairs.append(pair)
    row_offset = len(rows)
    for row in triple['rows']:
        rows.append(dict(row, arity=3))
    for p in triple['pairs']:
        pairs.append(dict(p, recipient=p['recipient']+row_offset, donor=p['donor']+row_offset, arity=3, cluster=p['cluster']+24))
    assert len(pairs) == 96 and len(rows) == 192
    old_prompts = set()
    for path in sorted(Path('runs').glob('*/panel.json')):
        data = json.loads(path.read_text())
        old_prompts.update(r['prompt'] for r in data.get('rows', []) if isinstance(r, dict) and 'prompt' in r)
    overlap = old_prompts & {r['prompt'] for r in rows}
    assert not overlap, sorted(overlap)[:2]
    panel_path = out/'CONFIRMATION_PANEL.json'
    panel_path.write_text(json.dumps(dict(rows=rows, pairs=pairs,
        scope='Two-term known operands in new contexts and new three-term questions, two formats each'), indent=2)+'\n')
    development_config = Path('configs/final_science03_qwen_profile_development.json')
    base = json.loads(development_config.read_text())
    configs = []
    for s, t in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]:
        c = copy.deepcopy(base)
        c.update(run_id=f'FS03_QWEN_FIELDS_CONFIRM_S{s}_T{t}_20260920', seeds=[s, t], batch_size=8,
                 max_length=96, budget_seconds=1500, candidate_family_frozen=True, evidence_level='frozen_confirmation',
                 scope='Frozen inference, new contexts and unseen three-term questions; dependent five-seed cycle',
                 budget='GPU lease, empirical generation throughput and13GBallocated bound')
        spec = c['source_field_evaluation']
        spec.update(source_seed=s, target_seeds=[t], panel=panel_path.as_posix(), evaluation_pairs=len(pairs),
                    balanced_forms=False, query_seed=209202603, partitions=4)
        if s == 1:
            spec['profile_cache'] = 'D:/CCAD_Storage/runs/final_science_20260920_round03/FS03_QWEN_PROFILE_DEVELOPMENT_20260920/source_profile.pt'
        path = Path(f'configs/final_science03_qwen_confirm_s{s}.json')
        assert not path.exists()
        path.write_text(json.dumps(c, indent=2)+'\n')
        configs.append(path)
    code = ['scripts/run_arithmetic_digit_components.py', 'scripts/arithmetic_source_fields.py',
            'src/ccad/source_field_inference.py', 'src/ccad/request_inference.py', 'src/ccad/intervention_transport.py',
            'scripts/analyze_qwen_field_confirmation.py', __file__]
    assets = [panel_path, triple_path, old_path, development_config,
        Path('D:/CCAD_Storage/runs/final_science_20260920_round03/FS03_QWEN_PROFILE_DEVELOPMENT_20260920/source_profile.pt')]
    for s, t in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]:
        assets.extend([Path(base['source_field_evaluation']['readout_run'])/f'readout_s{s}_t{t}.npz',
                       Path(base['source_field_evaluation']['relation_run'])/f'relation_s{s}_t{t}.npz'])
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), configurations=[identity(p) for p in configs],
        code=[identity(p) for p in code], inputs=[identity(p) for p in assets],
        primary='Balanced changed/unchanged exact source-answer agreement for eight unfitted masks, macro-average over operation and arity',
        primary_comparison='source-profile minus common-support Euclidean inference',
        secondary='Inherited relation/readout comparisons, full-answer fidelity and hybrid success, two forms and all five directed pairs retained',
        statistics='4000 paired resamples of48question clusters and4complementary-mask banks; five dependent SAE directions remain a fixed pool. Report each direction; no iid edge interval.',
        bootstrap_seed=2026092007,
        information='Profile and Euclidean receive exact source amplitudes. Readout estimates amplitudes from target codes. Profile1reused, profiles2to5 acquired by unchanged source-only recipe.',
        human_work=0, target_gradient_fits=0,
        two_term_scope='Known operand questions with new exact prompts; no claim of unseen two-term operands',
        three_term_scope='Unordered triples absent from all retained three-term panels; source gates and profile acquisition use original two-term fit pairs')
    (out/'CONFIRMATION_FREEZE.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(written_at_utc=result['written_at_utc'], configs=[str(p) for p in configs], pairs=len(pairs))))


if __name__ == '__main__':
    main()
