import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAYO_CONFIGS = ROOT / 'scripts' / 'mayo' / 'configs'
LOCAL_CONFIGS = ROOT / 'configs' / 'cross_seed_pilot'
ART = ROOT / 'artifacts' / 'containment'
M = '/mnt/nvme3-data/harvardhds/ccad-m364569'
# EXP-U Qwen readers moved to Mayo: the s2_t1 development readers and all confirmation readers. Executors, requests and
# documents equal those of the local configs; only paths differ, and the batches stay at the Mayo values that the
# 48 GB cards hold. The joint maps were trained locally and are copied to data/cross_seed/local_maps.
fresh = json.loads((ART / 'FRESH_PARADIGMS_QWEN.json').read_text())['included']
# s1_t2_ts2 joined after the local readers strained the 32 GB host memory of the workstation that runs them.
jobs = [('ureaders', 's1_t2', 2)] + [('ureaders', d, s) for d in ['s2_t1'] for s in [0, 1, 2]] + \
       [('cureaders', d, s) for d in ['s1_t2', 's2_t1'] for s in [0, 1, 2]]
for kind, direction, seed in jobs:
    base = json.loads((MAYO_CONFIGS / f'readers_qwen17_{direction}_ts{seed}.json').read_text())
    local = json.loads((LOCAL_CONFIGS / f'{kind}_qwen17_{direction}_ts{seed}.json').read_text())
    keep = {e['name']: e for e in base['executors']}
    u = f'{M}/data/cross_seed/local_maps/CSM_UJ_QWEN17_{direction}_ts{seed}_v1_20261011/maps.npz'
    cfg = dict(base, run_id=local['run_id'], purpose=local['purpose'],
               executors=[keep['hungarian_activation'], keep['membership_natural'], keep['rc_distparts'],
                          dict(name='u_distjoint', maps=u, key='membership_distjoint', cap=1.),
                          dict(name='s_statejoint', maps=u, key='membership_statejoint', cap=1.)])
    if kind == 'cureaders':
        cfg.update(heldout_paradigms=[f'{M}/data/grammar/fresh/{n}.jsonl' for n in fresh], natural_doc_offset=128,
                   natural_docs=local['natural_docs'])
    assert [(e['name'], e['key'], e['cap']) for e in cfg['executors']] == \
           [(e['name'], e['key'], e['cap']) for e in local['executors']]
    for k in ['random_requests', 'random_seed', 'part_names', 'pairs_per_paradigm', 'natural_docs', 'context',
              'eos_token_id', 'sentence_prefix', 'token_dtype']:
        assert cfg[k] == local[k], k
    assert cfg.get('natural_doc_offset', 0) == local.get('natural_doc_offset', 0)
    assert [Path(p).name for p in cfg['heldout_paradigms']] == [Path(p).name for p in local['heldout_paradigms']]
    (MAYO_CONFIGS / f'{kind}_qwen17_{direction}_ts{seed}.json').write_text(json.dumps(cfg, indent=1))
    print(cfg['run_id'], len(cfg['heldout_paradigms']), 'paradigms', cfg['natural_docs'], 'documents')
