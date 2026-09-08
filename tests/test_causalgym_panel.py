"""Check the actual leakage risk introduced by reciprocal external task pairs."""
from run_causalgym_native_transfer import prepare_panel


def test_shared_prompt_component_does_not_cross_splits():
    rows = []
    for i in range(24):
        a = ['<|endoftext|>', f'Name{i}', ' ran', ' because']
        b = ['<|endoftext|>', f'Other{i}', ' ran', ' because']
        c = ['<|endoftext|>', f'Third{i}', ' ran', ' because']
        for first, second, first_type, second_type in [(a, b, 'he', 'she'), (b, a, 'she', 'he'),
                                                      (b, c, 'she', 'he'), (c, b, 'he', 'she')]:
            rows.append(dict(base=first, src=second, base_label=' '+first_type, src_label=' '+second_type,
                             base_type=first_type, src_type=second_type, task='test'))
    cfg = dict(tasks=['test'], components_per_task=24, pairs_per_component=2, split_salt='test')
    panel, inventory = prepare_panel(rows, cfg)
    by_text = {}
    for row in panel:
        by_text.setdefault(row['text'], set()).add((row['component'], row['split']))
        assert panel[row['donor_id']]['donor_id'] == row['row_id']
    assert all(len(group)==1 for group in by_text.values())
    assert inventory['test']['connected_components'] == 24
    assert inventory['test']['unique_pairs'] == 48
    assert {row['split'] for row in panel} == {'fit', 'calibration', 'held_component_development'}
