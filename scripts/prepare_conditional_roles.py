import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from run_shift_explanation import source_groups


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-cell', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path('configs/morning_r05_restore_target_dev_v1.json').read_text())
    source = json.loads(Path(base['source_manifest']).read_text())
    groups, annotations = source_groups(base['notebook'], source['members'])
    sites = list(source['members'])
    nodes = []
    for si, site in enumerate(sites):
        for group, members in groups.items():
            chosen = members.get(site, [])
            if chosen:
                nodes.append(dict(name=f'{site}_{group}', site=site, site_index=si,
                                  group=group, members=chosen))

    def weights(node=None):
        return {site: [float(node is not None and site == node['site'] and member in node['members'])
                       for member in members] for site, members in source['members'].items()}

    queries = []
    for node in nodes:
        queries.append(dict(name='delete_'+node['name'], role='delete', node=node['name'],
                            weights=weights(node)))
        queries.append(dict(name='identity_'+node['name'], role='identity_control',
                            node=node['name'], weights=weights(), restore_weights=weights(node)))
    for early in nodes:
        for late in nodes:
            if early['site_index'] < late['site_index']:
                queries.append(dict(name='delete_'+early['name']+'_restore_'+late['name'],
                    role='restore', early_node=early['name'], restore_node=late['name'],
                    ablation_reference='delete_'+early['name'], weights=weights(early),
                    restore_weights=weights(late)))
    panel = json.loads(Path(base['evaluation_panel']).read_text())
    rows = []
    for label in [0, 1]:
        for gender in [0, 1]:
            bucket = sorted((r for r in panel['rows'] if (r['label'], r['gender']) == (label, gender)),
                            key=lambda r: r['document_sha256'])
            assert len(bucket) >= args.per_cell
            rows.extend(bucket[:args.per_cell])
    spec = dict(written_at_utc=datetime.now(timezone.utc).isoformat(), nodes=nodes,
                queries=queries, documents=[r['document_sha256'] for r in rows],
                context_split=['development']*len(rows), evidence_level='exposed_development',
                construction='All nonempty published annotation groups at each site, and all chronological pairs.',
                source_annotations=annotations)
    spec_path = args.output/'REQUESTS_DEVELOPMENT.json'
    save(spec_path, spec)
    base.update(run_id='RG05_conditional_roles_dev_seed2_v1_20260921',
        run_parent='REUSE_GENERALIZATION_05',
        run_storage_root='D:/CCAD_Storage/runs/reuse_generalization_20260921_round05',
        request_spec=str(spec_path), budget_seconds=900, eval_batch_size=8,
        purpose='Test whether source conditional effects predict target restoration choices.',
        scope=spec['construction'], evidence_level='exposed_development',
        budget='At most900seconds,13GB VRAM,2GB output; frozen dictionaries and relations.',
        extra_source_files=['scripts/prepare_conditional_roles.py'])
    save(args.output/'CONFIG_DEVELOPMENT.json', base)
    save(args.output/'DESIGN_IDENTITY.json', dict(
        requests_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        nodes=len(nodes), requests=len(queries), documents=len(rows)))
    print(json.dumps(dict(nodes=len(nodes), requests=len(queries), documents=len(rows))))


if __name__ == '__main__':
    main()
