from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]


def main():
    base_path = ROOT / 'configs/rg03_trajectory_development.json'
    base = json.loads(base_path.read_text())
    paths = []
    for name, small in [('smoke', True), ('development', False)]:
        config = dict(base,
            run_id=f'RG03_TRAJECTORY_SCOPE_{name.upper()}_T2_20260921',
            purpose='Measure which source locations are needed for trajectory feedback.',
            scope='Original development biographies and target 2. Source-active locations are defined without target outputs.',
            executions=['trajectory_feedback', 'trajectory_requested_sites', 'trajectory_active_tokens'],
            datasets=['human'], task_adapted_reference=None, query_progress=True,
            budget_seconds=600)
        if small:
            config.update(human_per_cell=1)
            config['human_queries'] = {k: v for k, v in config['human_queries'].items()
                                       if k in ['full', 'pronouns', 'names', 'associated_words']}
        path = ROOT / f'configs/rg03_trajectory_scope_{name}.json'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(config, indent=2) + '\n')
        paths.append(path)
    design = dict(
        written_at_utc=datetime.now(timezone.utc).isoformat(),
        comparison='The same feedback request is available at every original program site, at sites with a nonzero source request, or at tokens with a nonzero recorded source action.',
        scope='Two execution-location controls on the original 32-document development panel. Confirmation settings stay unchanged.',
        fixed='Source program, target 2 dictionaries, requests, maximum members, solver, candidate budget and data.',
        decision='If restricted feedback preserves the gain, the improved action target is useful within the corresponding source locations. If the gain requires broader locations, report that program realization uses compensatory target members.',
        checks='Eight-document smoke, nonnegative codes, member allowance, zero updates at excluded locations, and exact replay of unrestricted development predictions.',
        execution='Run after the three-target frozen confirmation and primary analysis. Preserve the original frozen executor before adding these modes.',
        identities={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [base_path, *paths]})
    path = ROOT / 'artifacts/reuse_generalization_20260921_round03/SCOPE_CONTROL_DESIGN.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(design, indent=2) + '\n')
    print(json.dumps(dict(configs=list(map(str, paths)), execution_ready=False)))


if __name__ == '__main__':
    main()
