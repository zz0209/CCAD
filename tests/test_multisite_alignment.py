import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_causalgym_multisite import aligned_positions


def test_unequal_changed_region_keeps_suffix_alignment():
    base = dict(tokens=[0,11,12,7,8,9],region_ends=[0,2,3,5],changed_region=1)
    donor = dict(tokens=[0,21,7,8,9],region_ends=[0,1,2,4],changed_region=1)
    assert aligned_positions(base,donor,'suffix_nonfinal')==[(2,1),(3,2),(4,3)]
    assert aligned_positions(base,donor,'suffix_all')==[(2,1),(3,2),(4,3),(5,4)]
    assert aligned_positions(base,donor,'full_tokens_equal_length') is None
    assert aligned_positions(base,donor,'region_ends_nonfinal')==[(0,0),(2,1),(3,2)]


def test_different_suffix_is_rejected_and_duplicate_positions_not_added():
    base = dict(tokens=[0,1,2,3],region_ends=[0,1,1,3],changed_region=1)
    donor = dict(tokens=[0,4,2,3],region_ends=[0,1,1,3],changed_region=1)
    assert aligned_positions(base,donor,'region_ends_nonfinal')==[(0,0),(1,1)]
    donor['tokens']=[0,4,9,3]
    try:
        aligned_positions(base,donor,'suffix_all')
    except ValueError as e:
        assert 'suffix' in str(e)
    else:
        raise AssertionError('Unaligned token suffix was accepted')
