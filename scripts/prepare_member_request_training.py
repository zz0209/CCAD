from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np

from prepare_source_column_reuse import save
from run_shift_explanation import source_groups


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/final_science_20260921_round05'
BULK = 'D:/CCAD_Storage/runs/final_science_20260921_round05'


def schedule(membership, steps, seed):
    assert np.all(membership.sum(1) == 1)
    rng = np.random.default_rng(seed)
    groups = membership.shape[1]
    alpha = rng.uniform(0, 1, (steps, groups))
    for step in range(0, steps, 2):
        bits = 1+(step//2)%((1 << groups)-1)
        alpha[step] = [(bits >> i) & 1 for i in range(groups)]
    semantic = alpha@membership.T
    independent = (rng.uniform(size=semantic.shape) < semantic).astype(float)
    balanced = semantic.copy()
    for step in range(1, steps, 2):
        for g in range(groups):
            ix = np.flatnonzero(membership[:, g])
            total = len(ix)*alpha[step, g]
            count = int(np.floor(total))
            values = np.zeros(len(ix))
            values[:count] = 1
            if count < len(ix):
                values[count] = total-count
            balanced[step, ix] = values[rng.permutation(len(ix))]
    independent[::2] = semantic[::2]
    assert np.allclose(balanced@membership, semantic@membership, atol=1e-12)
    values = dict(semantic=semantic, balanced=balanced, independent=independent)
    for q in values.values():
        assert np.all((q >= 0) & (q <= 1))
        assert np.array_equal(q[::2], semantic[::2])
    return values, alpha


def main():
    diagnostics = {}
    for setting in ['human', 'infinitive']:
        c = json.loads((ROOT/f'configs/fs04_{setting}_shared_columns_fit.json').read_text())
        if setting == 'human':
            members = json.loads(Path(c['source_manifest']).read_text())['members']
            groups, _ = source_groups(c['notebook'], members)
            p = np.array([[i in group.get(s, []) for group in groups.values()]
                          for s, ids in members.items() for i in ids], float)
            external = json.loads((ROOT/'artifacts/final_science_20260921_round04/SHARED_COLUMN_HUMAN_CONFIRMATION_PANEL.json').read_text())
            selected = []
            for profession in sorted({r['profession'] for r in external['rows']}):
                for gender in [0, 1]:
                    selected.extend([r for r in external['rows'] if r['profession'] == profession and r['gender'] == gender][:4])
            eval_path = OUT/'HUMAN_DEVELOPMENT_PANEL.json'
            save(eval_path, dict(rows=selected, evidence='Previously exposed round04 confirmation, now development'))
            c.update(evaluation_panel=str(eval_path), evaluation_per_cell=4, evaluation_split='dev',
                     program_evaluation_exclude_per_group=32, development_per_group=32,
                     evaluation_evidence='exposed_development',
                     evaluate_baselines=['input_initial', 'raw_reconstruction'])
            base = json.loads((ROOT/'configs/science04_shift_t2_v1.json').read_text())
            c['queries'] = [q for q in base['queries'] if not q.startswith('interior_') or int(q.split('_')[-1]) < 8]
        else:
            members = {'resid_4': [0, 1, 2, 3]}
            p = np.eye(2)[[0, 0, 0, 1]]
            external = json.loads((ROOT/'artifacts/final_science_20260921_round04/SHARED_COLUMN_INFINITIVE_CONFIRMATION_PANEL.json').read_text())
            eval_path = OUT/'INFINITIVE_DEVELOPMENT_PANEL.json'
            save(eval_path, dict(rows=external['rows'][:32], queries=external['queries'], answer=' to',
                                 evidence='Previously exposed round04 confirmation, now development'))
            c.update(panel=str(eval_path), baselines=['native_tangent_relation_8', 'raw_reconstruction'])
        c.update(variants=['tangent_mixed'], source_column_training_targets=[], evaluate_source_columns={},
                 run_parent='FINAL_SCIENCE_05', run_storage_root=BULK, training_seed=9153,
                 purpose='Test which member distinctions a fixed training budget preserves for subsequent intervention requests.',
                 scope='Same initial program, source contexts, losses, member allowance and updates; source-only request design changes.',
                 budget='512 updates per request design; original program and initial parameters; full retained development request panel.',
                 evidence_level='development', audit_opened=False)
        values, alpha = schedule(p, 512, 9153)
        diagnostics[setting] = dict(members=len(p), group_sizes=p.sum(0).astype(int).tolist(),
            rank={k: int(np.linalg.matrix_rank(v)) for k, v in values.items()},
            mean_change=float(np.abs((values['balanced']-values['semantic'])@p).max()))
        for family, requests in values.items():
            for smoke in [True, False]:
                stage = 'SMOKE' if smoke else 'DEV'
                steps = 16 if smoke else 512
                run_id = f'FS05_{setting.upper()}_{family.upper()}_{stage}_20260921'
                query_path = OUT/f'{setting.upper()}_{family.upper()}_{stage}_REQUESTS.json'
                offset = 0
                by_site = {}
                for site, ids in members.items():
                    by_site[site] = requests[:steps, offset:offset+len(ids)].tolist()
                    offset += len(ids)
                save(query_path, dict(members=members, family=family, seed=9153,
                     requests=by_site, group_weights=alpha[:steps].tolist(),
                     rule='Half semantic endpoints. Balanced interiors uniformly permute box vertices at fixed group mass; independent interiors use Bernoulli probabilities equal to group participation.',
                     scope='Nominal group mass is preserved by balanced requests; hidden-space and functional effect energies are measured outcomes.'))
                config = {**c, 'run_id': run_id, 'steps': steps,
                          'bulk_output_dir': f'{BULK}/{run_id}_checkpoints',
                          'training_request_panel': str(query_path), 'request_design': family,
                          'budget_seconds': 420 if smoke else 1200}
                if smoke and setting == 'human':
                    config['queries'] = ['pronouns', 'names', 'full', 'interior_00', 'member_subset_00']
                if smoke and setting == 'infinitive':
                    path = OUT/f'INFINITIVE_{family.upper()}_SMOKE_PANEL.json'
                    save(path, dict(rows=external['rows'][:8], queries={k: v for k, v in list(external['queries'].items())[:5]}, answer=' to'))
                    config['panel'] = str(path)
                save(ROOT/f'configs/fs05_{setting}_{family}_{stage.lower()}.json', config)
    save(OUT/'REQUEST_DESIGN.json', dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
         diagnostics=diagnostics, hypothesis='Training independently varying members improves later member requests while retaining semantic group responses.',
         comparison='Semantic, fixed-group-mass and independent-member requests under the same program and response loss.',
         rule='All scientific confirmation uses new data after development selection. Coordinate rank alone is a source-side design check.'))
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == '__main__':
    main()
