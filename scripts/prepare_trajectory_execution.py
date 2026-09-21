from pathlib import Path
from datetime import datetime, timezone
import json


ROOT=Path(__file__).resolve().parents[1]


def main():
    base=json.loads((ROOT/'configs/rg03_request_realization_development.json').read_text())
    for name,small in [('smoke',True),('development',False)]:
        c=dict(base,run_id=f'RG03_TRAJECTORY_{name.upper()}_T2_20260921',
            purpose='Separate local source actions from accumulated trajectory correction in a complete explanation.',
            scope='One additional source-program pass per requested input supplies reference states. No target responses, fitting or dictionary updates. Previously exposed development.',
            executions=['inverse_budget','trajectory_action','trajectory_feedback','raw_readout'],
            query_progress=False)
        if small:
            c.update(human_per_cell=1,grammar_rows=4,budget_seconds=600,task_adapted_reference=None)
            c['human_queries']={k:v for k,v in c['human_queries'].items() if k in ['full','center','pronouns']}
        path=ROOT/f'configs/rg03_trajectory_{name}.json'
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(c,indent=2)+'\n')
    path=ROOT/'artifacts/reuse_generalization_20260921_round03/TRAJECTORY_DESIGN.json'
    path.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Feedback against source-program intermediate states reduces propagation of native realization error across sites.',
        prediction='The two trajectory constructions agree with request-specific inversion in a single-site explanation; only multi-site execution can benefit from feedback.',
        information='An extra source-program pass is required for every requested input. Targets remain frozen, and target gradients or labels are never queried.',
        interpretation='Feedback realizes the cumulative source request across sites. Local member identities are not assumed to remain the same functional intervention at each site.',
        decision='A large full-program gain and single-site equivalence justify fresh-context and target confirmation; otherwise retain it as an evaluated mechanism.'),indent=2)+'\n')


if __name__=='__main__':
    main()
