"""Describe the finite test-menu ceiling; do not fit or deploy an oracle.

The saved policies choose one candidate per query before test evaluation.
This post-outcome diagnostic separates candidate-menu limitations from
shortlisting and final-choice regret on the same observed test rows.
"""
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixed', default='compiled_functional_axis')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    started = datetime.now(timezone.utc).isoformat()
    timer, cpu = time.perf_counter(), time.process_time()
    records, policies, inputs = [], [], []
    for supplied in args.runs:
        run = supplied.resolve()
        cfg = json.loads((run/'config.resolved.json').read_text())
        assert json.loads((run/'status.json').read_text())['status'] == 'PASS'
        assert cfg['evaluation_split'] == 'test', 'This report is for saved test queries.'
        choices = json.loads((run/'selection_choices.json').read_text())['choices']
        assert len(choices) == len(cfg['tasks'])*len(cfg['objectives'])*len(cfg['seed_pairs'])
        # Directly replay every saved query/method IIA from its original rows.
        raw = defaultdict(lambda: [0, 0])
        with (run/'metrics.raw.jsonl').open() as stream:
            for line in stream:
                r = json.loads(line)
                values = raw[(r['query'], r['method'])]
                values[0] += int(r['iia'])
                values[1] += 1
        model = cfg['base_model'] if 'base_model' in cfg else (
            'GPT2Medium' if 'gpt2' in run.name else 'Pythia1B' if 'pythia' in run.name else None)
        assert isinstance(model, str) and model
        for c in choices:
            y = {n: v['iia'] for n, v in c['held_summary'].items()}
            for n, value in y.items():
                successes, count = raw[(c['query'], n)]
                assert count == cfg.get('test_rows', 100) and abs(successes/count-value) < 1e-12
            candidates = sorted(c['candidate_stats'])
            assert args.fixed in candidates
            ceiling = max(y[n] for n in candidates)
            assert abs(ceiling-c['oracle_native_held_iia']) < 1e-12
            best = [n for n in candidates if abs(y[n]-ceiling) < 1e-12]
            row = dict(run=run.name, model=model, objective=c['objective'],
                       query=c['query'], task=c['task'], source_seed=c['source_seed'],
                       target_seed=c['target_seed'], candidates=len(candidates),
                       fixed_iia=y[args.fixed], observed_menu_oracle_iia=ceiling,
                       headroom=ceiling-y[args.fixed], oracle_methods=best,
                       raw_signed_iia=y['raw_signed_distill'], raw_das_iia=y['raw_das'],
                       raw_signed_minus_oracle=y['raw_signed_distill']-ceiling,
                       raw_das_minus_oracle=y['raw_das']-ceiling)
            records.append(row)
            for b in c['budget_choices']:
                shortlist = b.get('shortlist', candidates)
                assert set(shortlist) <= set(candidates) and b['selected'] in shortlist
                shortlist_max = max(y[n] for n in shortlist)
                screen_loss, choice_loss = ceiling-shortlist_max, shortlist_max-y[b['selected']]
                assert abs(screen_loss+choice_loss-(ceiling-y[b['selected']])) < 1e-12
                policies.append(dict(run=run.name, model=model, objective=c['objective'],
                    query=c['query'], task=c['task'], source_seed=c['source_seed'], target_seed=c['target_seed'],
                    policy=b['policy'], budget=b['budget'], selected=b['selected'],
                    selected_iia=y[b['selected']], gain_over_fixed=y[b['selected']]-y[args.fixed],
                    regret=ceiling-y[b['selected']], shortlist_regret=screen_loss, refinement_regret=choice_loss,
                    fixed_in_shortlist=args.fixed in shortlist, fixed_selected=b['selected']==args.fixed))
        for name in ['config.resolved.json', 'status.json', 'selection_choices.json', 'metrics.raw.jsonl']:
            p = run/name
            inputs.append(dict(path=p.relative_to(root).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    conditions = []
    for model, obj in sorted({(r['model'], r['objective']) for r in records}):
        rows = [r for r in records if (r['model'], r['objective']) == (model, obj)]
        ps = [r for r in policies if (r['model'], r['objective']) == (model, obj)]
        aggregated = []
        for name, budget in sorted({(r['policy'], r['budget']) for r in ps}):
            rr = [r for r in ps if (r['policy'], r['budget']) == (name, budget)]
            aggregated.append(dict(policy=name, budget=budget, queries=len(rr),
                **{k: mean(r[k] for r in rr) for k in ['selected_iia', 'gain_over_fixed', 'regret', 'shortlist_regret', 'refinement_regret']},
                fixed_selected=sum(r['fixed_selected'] for r in rr), fixed_in_shortlist=sum(r['fixed_in_shortlist'] for r in rr),
                selected_counts=dict(Counter(r['selected'] for r in rr))))
        conditions.append(dict(model=model, objective=obj, queries=len(rows),
            **{k: mean(r[k] for r in rows) for k in ['fixed_iia', 'observed_menu_oracle_iia', 'headroom', 'raw_signed_iia', 'raw_das_iia', 'raw_signed_minus_oracle', 'raw_das_minus_oracle']},
            fixed_attains_oracle=sum(r['headroom'] < 1e-12 for r in rows),
            strictly_better_queries=[r for r in rows if r['headroom'] > 1e-12], policies=aggregated))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('queries', records), ('policies', policies)]:
        with (args.output/(name+'.csv')).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = dict(started_at_utc=started, written_at_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-timer, process_cpu_seconds=time.process_time()-cpu,
        fixed_candidate=args.fixed, conditions=conditions, query_records=records, input_files=inputs,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Post-outcome descriptive diagnostic of the unchanged observed finite candidate menu, choosing one candidate per query. The maximum uses test outcomes and is not a deployable selector, a population upper bound, a bound on per-example adaptive policies, or a new confirmation endpoint. Shared tasks and seed nodes are dependent. The fixed reference was the frozen method under confirmation; no budget, candidate or test row was selected for a main claim. All original raw method IIA values were replayed exactly.')
    (args.output/'SUMMARY.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['conditions', 'query_records', 'input_files']}))
    for c in conditions:
        print(json.dumps({k:v for k,v in c.items() if k not in ['policies', 'strictly_better_queries']}))


if __name__ == '__main__':
    main()
