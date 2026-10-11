import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'configs' / 'cross_seed_pilot'
ART = ROOT / 'artifacts' / 'containment'
BLIMP = ROOT / 'artifacts' / 'scientific_reform_20260913' / 'reference_material' / 'blimp' / '3e56b06fcabca9b30822fc66435fca6b1aa40bb1'
# EXP-U confirmation readers (research lead decision 4): the paradigms that the source-only filter admitted for each
# model, and generic-text documents after the 128 that development readers and transfer evaluations used (GPT-2 has
# 128 such documents, Qwen 92). Executors are those of the development readers.
for model, prefix, docs, runs in [('GPT2', 'ureaders_gpt2_', 128, ['s1_t2', 's1_t3', 's1_t4', 's1_t5', 's2_t3', 's3_t4', 's4_t5', 's5_t1']),
                                  ('QWEN', 'ureaders_qwen17_', 92, [f'{d}_ts{s}' for d in ['s1_t2', 's2_t1'] for s in [0, 1, 2]])]:
    included = json.loads((ART / f'FRESH_PARADIGMS_{model}.json').read_text())['included']
    for r in runs:
        cfg = json.loads((OUT / f'{prefix}{r}.json').read_text())
        tag = 'GPT2' if model == 'GPT2' else 'QWEN17'
        cfg.update(run_id=f'CSM_CREADERS_{tag}_{r}_v1_20261011', heldout_paradigms=[str(BLIMP / f'{n}.jsonl') for n in included],
                   natural_doc_offset=128, natural_docs=docs,
                   purpose='READER_COVERAGE EXP-U confirmation: fresh BLiMP paradigms admitted by the source model and unused generic text.')
        (OUT / f'c{prefix}{r}.json').write_text(json.dumps(cfg, indent=1))
        print(cfg['run_id'], len(included), 'paradigms')
