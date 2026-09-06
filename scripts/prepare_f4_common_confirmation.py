"""Apply the frozen common-panel methods to fresh inputs; never fit or select outcomes."""
import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from run_f4_source_reference_causal import ROOT, write, sha256
from run_r011s1_raw_hook_asset import aggregate, entry
from ccad.artifacts import validate_run_directory

CORPUS = 'F4_common_confirmation_corpus_v1_20260906'
CODES = 'F4_common_confirmation_codes_v1_20260906'
METHODS = ['global_matching_geometric', 'global_matching_pair_calibrated']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked(spec):
    path = ROOT / spec['path']
    if sha256(path) != spec['sha256']:
        raise ValueError('Frozen input changed: ' + str(path))
    return path


def new_config(path, cfg):
    path = ROOT / path
    if path.exists():
        raise FileExistsError(path)
    write(path, cfg)
    print(json.dumps({'config': str(path), 'sha256': sha256(path)}), flush=True)


def source_config(panel, corpus_cfg):
    checked(dict(path=panel['source_preparation_template'], sha256=panel['source_preparation_template_sha256']))
    cfg = read(ROOT / panel['source_preparation_template'])
    for key in ['readout_ablation', 'frozen_policy']:
        cfg.pop(key, None)
    bulk = Path('D:/CCAD_Storage/paired_assets') / CODES
    cfg.update(run_id=f"F4_common_confirmation_source_{panel['label']}_v1_20260906",
               methods=['target', 'raw'], source_query_subset=panel['fixed_queries'],
               bulk_asset_dir=str(bulk), raw_hook_asset_dir=str(bulk),
               asset_manifest_sha256=sha256(bulk / 'asset_manifest.json'),
               raw_hook_manifest_sha256=sha256(bulk / 'raw_hook_manifest.json'), paired_corpus_run=CORPUS,
               resource_lease='cpu-heavy resource_manager.run', resource_lease_reason='Bounded source-code scans and local tokenizer; no GPU or endpoint forward',
               frozen_corpus_config_sha256=sha256(corpus_cfg), budget='Frozen source-only preparation,0LM; both panels/map application<=120s CPU.')
    for key, name in [('token_manifest', 'token_manifest.json'), ('sequence_records', 'sequence_records.json')]:
        path = ROOT / 'runs' / CORPUS / 'artifacts' / name
        cfg[key + '_path'] = path.relative_to(ROOT).as_posix()
        cfg[key + '_sha256'] = sha256(path)
    return cfg


def coordinate(ix, acts, choice, matched, beta, length, width):
    positions = np.asarray(choice['intervention_positions'], dtype=int)
    donor_positions = np.asarray(choice['donor_positions'], dtype=int)
    rows = positions + choice['sequence'] * length
    donors = donor_positions + choice['donor_sequence'] * length
    z = np.zeros((len(rows), width)); zd = np.zeros_like(z)
    np.add.at(z, (np.arange(len(rows))[:, None], ix[rows]), acts[rows])
    np.add.at(zd, (np.arange(len(rows))[:, None], ix[donors]), acts[donors])
    value = np.zeros((length, 1))
    value[positions, 0] = (z - zd)[:, matched] @ beta
    return value


def prepare_consumers(cfg, config_path):
    run = ROOT / 'runs/F4_common_confirmation_apply_v1_20260906'
    run.mkdir(exist_ok=False); start = time.perf_counter()
    write(run / 'config.resolved.json', cfg)
    sources = [Path(__file__), ROOT / 'scripts/run_f4_source_reference_causal.py',
               ROOT / 'scripts/run_r011s1_raw_hook_asset.py', ROOT / 'src/ccad/artifacts.py']
    code = []
    for path in sources:
        rel = path.relative_to(ROOT).as_posix(); dest = run / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(path.read_bytes())
        code.append(dict(path=rel, sha256=sha256(path), bytes=path.stat().st_size, snapshot_path='source_snapshot/' + rel))
    write(run / 'code_hashes.json', dict(files=code, aggregate_sha256=aggregate(code), snapshot_root='source_snapshot'))
    write(run / 'manifest.json', dict(schema_version='fcc.frozen.application.v1', run_id=run.name,
          run_parent='F4', milestone='M4', purpose='Fixed global group readouts on source-selected fresh cases',
          evidence_level='fresh_document_frozen_map_application', started_utc=datetime.now(timezone.utc).isoformat(),
          project_root=str(ROOT), config_hash=sha256(run / 'config.resolved.json'), code_snapshot_hash=aggregate(code),
          source_snapshot_required=True, audit_opened=False, candidate_family_frozen=True,
          mean_constants_source_split='original independent mean; cancels in donor differences',
          threshold_source_split='frozen development source rule', statistics_unit='dependent document/query/seed',
          device='CPU', seeds=[1, 2, 3, 4, 5], resource_lease='cpu-heavy resource_manager.run',
          resource_lease_reason='Bounded sparse-code identity checks and fixed coordinate application'))
    write(run / 'status.json', dict(status='RUNNING')); (run / 'stderr.log').write_text('')
    inputs = []; seen = set(); rows = []; error = None; checks = {}
    def consume(path, digest=None):
        path = ROOT / path
        if path not in seen:
            inputs.append(entry(path, 'CCAD frozen confirmation input', 'application_input')); seen.add(path)
        if digest and sha256(path) != digest: raise ValueError('Application input changed: ' + str(path))
        return path
    try:
        frozen = cfg['frozen_scope']; consume(config_path)
        for spec in frozen['code_identities']: consume(spec['path'], spec['sha256'])
        coefficients = np.load(consume(frozen['matching_coefficients']['path'], frozen['matching_coefficients']['sha256']), allow_pickle=False)
        consume(frozen['matching_details']['path'], frozen['matching_details']['sha256'])
        length = cfg['context_length']; width = 3072; sparse = {}
        def open_codes(tag, bulk):
            manifest = read(consume(bulk / 'asset_manifest.json'))
            part = next(p for p in manifest['splits'] if p['split'] == 'calibration')
            for seed in range(1, 6):
                arrays = {}
                for item in part['files']:
                    if item['seed'] == seed:
                        path = consume(item['path'], item['sha256'])
                        arrays[item['dtype']] = np.memmap(path, dtype='<u2' if item['dtype']=='uint16' else '<f4', mode='r', shape=tuple(item['shape']))
                sparse[tag, seed] = arrays['uint16'], arrays['float32']
            return manifest
        # Replay the actual old coordinate interface before using it on fresh inputs.
        oldbase = read(checked(dict(path=frozen['panels'][0]['base_causal_config'], sha256=frozen['panels'][0]['base_causal_config_sha256'])))
        open_codes('old', Path(oldbase['bulk_asset_dir']))
        oldrun = ROOT / 'runs/F4_global_matching_prepare_v1_20260906'
        oldarrays = np.load(consume(oldrun / 'candidate_coordinates.npz'), allow_pickle=False)
        replay = []
        for label in ['original', 'expanded']:
            for record in read(consume(oldrun / f'{label}_candidate_index.json'))['rows']:
                key = f"s{record['source_seed']}_a{record['source_atom']}_t{record['target_seed']}"
                name = record['method'].removeprefix('global_matching_')
                result = coordinate(*sparse['old', record['target_seed']], record, coefficients[key+'_matched_atoms'], coefficients[key+'_'+name], length, width)
                replay.append(float(np.max(np.abs(result - oldarrays[record['array_key']]))))
        assert len(replay) == 192 and max(replay) == 0
        bulk = Path('D:/CCAD_Storage/paired_assets') / CODES
        open_codes('new', bulk); arrays = {}; output = []; total_cases = 0
        for panel in frozen['panels']:
            label = panel['label']; source_run = ROOT / 'runs' / f'F4_common_confirmation_source_{label}_v1_20260906'
            assert read(consume(source_run / 'status.json'))['status'] == 'PASS'
            payload = read(consume(source_run / 'matching.json')); base = read(consume(source_run / 'config.resolved.json'))
            assert [(r['source_seed'],r['source_atom']) for r in payload['choices'][::2]] == [tuple(q) for q in panel['fixed_queries']]
            identities = []; records = []
            for choice in payload['choices']:
                e = choice['entry']
                if e is None: continue
                assert choice['source_scope']['supported']
                s, a = choice['source_seed'], choice['source_atom']; total_cases += 1
                identities.append(dict(source_seed=s, source_atom=a, **{k:e[k] for k in ['condition','sequence','donor_sequence']}))
                for t in range(1,6):
                    if t == s: continue
                    key = f's{s}_a{a}_t{t}'
                    for method in METHODS:
                        name = method.removeprefix('global_matching_'); array_key = f'coordinate_{len(arrays)}'
                        arrays[array_key] = coordinate(*sparse['new',t], e, coefficients[key+'_matched_atoms'], coefficients[key+'_'+name], length, width)
                        records.append(dict(source_seed=s,source_atom=a,target_seed=t,method=method,array_key=array_key,**e))
            base.pop('source_preparation_only'); base.update(run_id=f'F4_common_confirmation_{label}_v1_20260906',
                methods=frozen['methods'], saved_atom_families=[panel['saved_atom']], expected_evaluated_cases=len(identities),
                frozen_evaluated_requests=identities, probability_endpoints=frozen['probability_endpoints'],
                evidence_level='fresh_document_confirmation_frozen_common_panel', scope_limit=cfg['scope_limit']+' '+frozen['unequal_capacity'],
                case_replay=dict(path=(source_run/'matching.json').relative_to(ROOT).as_posix(),sha256=sha256(source_run/'matching.json'),selected_only=False,source_selection_scope='all_supported',export_details=False),
                resource_lease='cpu-heavy -> gpu-0 resource_manager.run', resource_lease_reason='Actual bounded inference; no whole-disk lease',
                budget=f'{len(identities)}cases*(3+4*5)={len(identities)*23}LM; both panels<=736LM/600s wall;0refit.')
            index = run / f'{label}_candidate_index.json'
            write(index, dict(rows=records, factors_sha256=base['factors_sha256'], surface_sha256=base['surface_sha256'],
                  sequence_records_sha256=base['sequence_records_sha256'], case_selection_sha256=base['case_replay']['sha256'], scale='unscaled_source_basis_coordinates'))
            output.append((label, base, index)); rows.append(dict(panel=label,requests=len(payload['choices']),matched=len(identities),selected=sum(bool(r['entry'] and r['source_scope']['selected']) for r in payload['choices'])))
        arraypath = run / 'candidate_coordinates.npz'; np.savez_compressed(arraypath, **arrays)
        for label, base, index in output:
            base['saved_candidate_coordinates'] = dict(index_path=index.relative_to(ROOT).as_posix(),index_sha256=sha256(index),arrays_path=arraypath.relative_to(ROOT).as_posix(),arrays_sha256=sha256(arraypath),methods=METHODS)
            new_config(f'configs/f4_common_confirmation_{label}_v1.json', base)
        checks = dict(old192coordinate_replay_exact=max(replay)==0,all32requests=sum(r['requests'] for r in rows)==32,
                      all_matched_coordinates=len(arrays)==total_cases*8,finite=all(np.isfinite(a).all() for a in arrays.values()),no_refit=True,no_model_forward=True,audit_closed=True)
    except Exception:
        error = traceback.format_exc(); (run/'stderr.log').write_text(error,encoding='utf-8')
    write(run/'inputs.json',dict(inputs=inputs))
    (run/'metrics.raw.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    status = 'PASS' if error is None and checks and all(checks.values()) else 'FAIL'
    summary = dict(status=status,error=error,checks=checks,model_forwards=0,wall_seconds=time.perf_counter()-start,panels=rows,metrics_raw_sha256=sha256(run/'metrics.raw.jsonl'))
    write(run/'metrics.summary.json',summary);write(run/'stdout.log',summary)
    write(run/'environment.json',dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,platform=platform.platform(),torch='not_used',transformers='not_used',cuda='not_used',gpu='not_used'))
    write(run/'status.json',dict(status=status,error=error))
    valid=validate_run_directory(run);write(run/'contract_validation.json',dict(ok=valid.ok,errors=list(valid.errors)))
    print(json.dumps(dict(summary=summary,contract_ok=valid.ok,errors=list(valid.errors))),flush=True)
    return int(status!='PASS' or not valid.ok)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['codes','source','consumers'],required=True);args=parser.parse_args()
    config_path=ROOT/'configs/f4_common_confirmation_corpus_v1.json';cfg=read(config_path)
    for spec in cfg['frozen_scope']['code_identities']:checked(spec)
    assert read(ROOT/'runs'/CORPUS/'status.json')['status']=='PASS'
    if args.stage=='codes':
        code=read(ROOT/'configs/f4_compact_component_confirmation_codes_v1.json')
        code.update(run_id=CODES,bulk_output_dir=str(Path('D:/CCAD_Storage/paired_assets')/CODES),paired_corpus_run=CORPUS,
                    purpose='One shared hook pass on fresh common functional-confirmation inputs; five frozen SAE codes, no endpoint fitting',frozen_corpus_config_sha256=sha256(config_path),
                    token_manifest_path=f'runs/{CORPUS}/artifacts/token_manifest.json',token_manifest_sha256=sha256(ROOT/'runs'/CORPUS/'artifacts/token_manifest.json'))
        new_config('configs/f4_common_confirmation_codes_v1.json',code)
    elif args.stage=='source':
        assert read(ROOT/'runs'/CODES/'status.json')['status']=='PASS'
        for panel in cfg['frozen_scope']['panels']:new_config(f"configs/f4_common_confirmation_source_{panel['label']}_v1.json",source_config(panel,config_path))
    else:return prepare_consumers(cfg,config_path)
    return 0


if __name__=='__main__':raise SystemExit(main())
