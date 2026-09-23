import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np


REQUESTS = ['whole1', 'whole_half', 'H1', 'H2', 'union', 'complement']
METHODS = ['source1', 'source2', 'graph', 'PW_selected', 'raw']
SCALARS = ['weighted_response_error', 'source_response_energy', 'response_energy',
           'next_token_logprob', 'clean_next_token_logprob', 'source_next_token_logprob',
           'argmax', 'clean_argmax', 'source_argmax', 'kl_clean_to_operation',
           'kl_source_to_operation', 'edit_norm']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(path):
    path = Path(path)
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return dict(path=str(path.resolve()), sha256=digest)


def load_run(run):
    assert read(run/'status.json')['status'] == 'PASS'
    index, panel, config = read(run/'response_index.json'), read(run/'panel.json'), read(run/'config.resolved.json')
    rows = panel['rows']
    jobs, baseline = {}, np.full((len(rows), 1024), np.nan, np.float32)
    for record in index['chunks']:
        with np.load(record['path']) as saved:
            assert str(saved['identity']) == index['identity'] and str(saved['status']) == 'PASS'
            ids = saved['row_indices'].astype(np.int64)
            assert len(ids) == record['count']
            if record['kind'] == 'baseline':
                assert np.isnan(baseline[ids]).all()
                baseline[ids] = saved['ln_f']
                continue
            key = (record['mechanism'], record['method'], record['operation'])
            if key not in jobs:
                jobs[key] = dict(rows=[], ln_f=[], members=saved['member_ids'].tolist(),
                    fraction=float(saved['fraction']), kind=record['kind'],
                    values={name: [] for name in SCALARS})
            job = jobs[key]
            assert job['members'] == saved['member_ids'].tolist()
            assert job['fraction'] == float(saved['fraction'])
            job['rows'].append(ids)
            job['ln_f'].append(saved['ln_f'])
            for name in SCALARS:
                job['values'][name].append(saved[name])
    assert np.isfinite(baseline).all()
    for key, job in jobs.items():
        ids = np.concatenate(job['rows'])
        order = np.argsort(ids)
        assert len(np.unique(ids)) == len(ids)
        job['rows'] = ids[order]
        job['ln_f'] = np.concatenate(job['ln_f'])[order]
        job['values'] = {name: np.concatenate(values)[order].astype(np.float64)
                         for name, values in job['values'].items()}
        assert all(np.isfinite(values).all() for values in job['values'].values())
    maps = {mechanism: read(Path(config['target_map_run'])/(mechanism+'_target_group.json'))
            for mechanism in config['mechanisms']}
    return dict(index=index, panel=panel, config=config, jobs=jobs, baseline=baseline, maps=maps,
        input=dict(run=str(run.resolve()), index=identity(run/'response_index.json'),
                   panel=identity(run/'panel.json'), configuration=identity(run/'config.resolved.json')))


def estimate(point, samples):
    valid = np.isfinite(samples)
    return dict(value=float(point) if np.isfinite(point) else None,
                ci95=np.quantile(samples[valid], [.025, .975]).tolist() if valid.any() else None,
                valid_bootstrap=int(valid.sum()))


def ratio_root(numerator, denominator):
    result = np.full_like(np.asarray(denominator, dtype=float), np.nan)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    np.sqrt(np.divide(numerator, denominator, out=np.zeros_like(result), where=valid),
            out=result, where=valid)
    return result


def statistics(values, weights):
    error = weights@values['weighted_response_error']
    source = weights@values['source_response_energy']
    return dict(nrmse=ratio_root(error, source), error_squared=error, source_energy=source,
        error_rms=np.sqrt(error), source_rms=np.sqrt(source),
        response_rms=np.sqrt(weights@values['response_energy']),
        next_token_logprob=weights@values['next_token_logprob'],
        next_token_logprob_change=weights@(values['next_token_logprob']-values['clean_next_token_logprob']),
        next_token_logprob_difference_from_source=weights@(values['next_token_logprob']-values['source_next_token_logprob']),
        argmax_preservation=weights@(values['argmax']==values['clean_argmax']),
        argmax_source_agreement=weights@(values['argmax']==values['source_argmax']),
        kl_clean_to_operation=weights@values['kl_clean_to_operation'],
        kl_source_to_operation=weights@values['kl_source_to_operation'],
        edit_rms=np.sqrt(weights@np.square(values['edit_norm'])))


def document_weights(rows, strata, document_ids, multiplicity):
    local_documents = np.array([document_ids[row['document_sha256']] for row in rows])
    position_counts = multiplicity[:, local_documents].astype(np.float64)
    weights = np.zeros_like(position_counts)
    valid = np.ones(len(multiplicity), dtype=bool)
    cell_counts = {}
    for stratum in strata:
        select = np.array([row['stratum'] == stratum for row in rows])
        assert select.any(), 'Declared stratum absent from the actual panel'
        cell_counts[stratum] = dict(positions=int(select.sum()),
            documents=len(set(local_documents[select].tolist())))
        denominator = position_counts[:, select].sum(1)
        valid &= denominator > 0
        weights[:, select] = np.divide(position_counts[:, select], denominator[:, None],
            out=np.zeros_like(position_counts[:, select]), where=denominator[:, None] > 0)/len(strata)
    # 缺层重复不参与该机制的区间计算，保留实际数量且不追加抽样。
    weights[~valid] = np.nan
    return weights, valid, cell_counts


def analyze(runs, output, draws, seed):
    start = time.perf_counter()
    assert not output.exists()
    loaded = [load_run(run) for run in runs]
    all_documents = sorted({row['document_sha256'] for run in loaded for row in run['panel']['rows']})
    document_ids = {value: index for index, value in enumerate(all_documents)}
    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(len(all_documents), np.full(len(all_documents), 1/len(all_documents)), size=draws)
    multiplicity = np.concatenate([np.ones((1, len(all_documents)), dtype=np.int64), multiplicity])
    summaries, contrasts, member_tables, union_checks, position_arrays, cohorts = {}, {}, {}, {}, {}, {}
    statistics_samples = {}
    for run in loaded:
        for mechanism in run['config']['mechanisms']:
            assert mechanism not in cohorts, 'One completed run per mechanism in this analysis'
            reference = run['jobs'][mechanism, 'source1', 'whole1']
            row_indices = reference['rows']
            rows = [run['panel']['rows'][i] for i in row_indices]
            strata = run['panel']['stratum_order']
            assert len(strata) == 4 and len(set(strata)) == 4
            weights, valid, cells = document_weights(rows, strata, document_ids, multiplicity)
            assert valid[0]
            cohorts[mechanism] = dict(positions=len(rows), documents=len({r['document_sha256'] for r in rows}),
                strata=cells, stratum_order=strata, valid_bootstrap=int(valid[1:].sum()),
                empty_stratum_bootstrap=int((~valid[1:]).sum()),
                confidence_scope='Paired document-cluster percentile intervals among draws containing all four strata; omitted draws retained in counts, no replacement draws')
            position_arrays[mechanism+'__bootstrap_valid'] = valid[1:]
            position_arrays[mechanism+'__document_sha256'] = np.array([r['document_sha256'] for r in rows])
            position_arrays[mechanism+'__stratum'] = np.array([r['stratum'] for r in rows])
            position_arrays[mechanism+'__row_id'] = np.array([str(r['row_id']) for r in rows])
            local_stats = {}
            for (objective, method, request), job in run['jobs'].items():
                if objective != mechanism:
                    continue
                np.testing.assert_array_equal(job['rows'], row_indices)
                own_reference = run['jobs'][mechanism, 'source1', request]
                np.testing.assert_allclose(job['values']['source_response_energy'],
                    own_reference['values']['source_response_energy'], rtol=1e-12, atol=1e-12)
                values = statistics(job['values'], weights)
                key = mechanism+'/'+method+'/'+request
                local_stats[method, request] = values
                statistics_samples[key] = values
                summaries[key] = {name: estimate(value[0], value[1:]) for name, value in values.items()}
                for name, value in job['values'].items():
                    position_arrays['__'.join((mechanism, method, request, name))] = value
            for request in REQUESTS:
                for first, second in [('graph', 'PW_selected'), ('graph', 'raw'),
                                       ('graph', 'source2'), ('PW_selected', 'raw')]:
                    contrast = mechanism+'/'+first+'_minus_'+second+'/'+request
                    contrasts[contrast] = {}
                    for metric in ['nrmse', 'error_rms', 'next_token_logprob', 'argmax_preservation', 'kl_source_to_operation']:
                        delta = local_stats[first, request][metric]-local_stats[second, request][metric]
                        contrasts[contrast][metric] = estimate(delta[0], delta[1:])
            singleton_names = sorted([request for method, request in local_stats if method == 'PW_source1'],
                                     key=lambda name: int(name.removeprefix('member_')))
            assert singleton_names
            member_rows, member_error, member_source = [], [], []
            for request in singleton_names:
                job = run['jobs'][mechanism, 'PW_source1', request]
                member_rows.append(dict(source_member=int(request.removeprefix('member_')),
                    target_member=job['members'][0], statistics=summaries[mechanism+'/PW_source1/'+request]))
                member_error.append(job['values']['weighted_response_error'])
                member_source.append(job['values']['source_response_energy'])
            pooled_error = weights@np.sum(member_error, axis=0)
            pooled_source = weights@np.sum(member_source, axis=0)
            pooled_ratio = ratio_root(pooled_error, pooled_source)
            member_tables[mechanism] = dict(rows=member_rows,
                energy_aggregated_nrmse=estimate(pooled_ratio[0], pooled_ratio[1:]),
                summed_member_error=estimate(pooled_error[0], pooled_error[1:]),
                summed_member_source_energy=estimate(pooled_source[0], pooled_source[1:]),
                normalization='Square root of sum of equally-stratified singleton squared errors divided by sum of each singleton own source response energy',
                PW_selected_source_seed=run['maps'][mechanism]['PW_selected_source_seed'],
                same_PW_correspondence_as_group=run['maps'][mechanism]['PW_selected_source_seed']==1)
            source_change = reference['ln_f'].astype(np.float64)-run['baseline'][row_indices]
            source_state_energy = weights@np.sum(source_change**2, axis=1)
            union_checks[mechanism] = {}
            for method in METHODS:
                whole = run['jobs'][mechanism, method, 'whole1']
                union = run['jobs'][mechanism, method, 'union']
                difference = union['ln_f'].astype(np.float64)-whole['ln_f'].astype(np.float64)
                state_squared = np.sum(difference**2, axis=1)
                squared = weights@state_squared
                normalized = ratio_root(squared, source_state_energy)
                primary_difference = local_stats[method, 'union']['nrmse']-local_stats[method, 'whole1']['nrmse']
                union_checks[mechanism][method] = dict(whole_members=len(whole['members']),
                    union_members=len(union['members']),
                    union_only_members=sorted(set(union['members'])-set(whole['members'])),
                    whole_only_members=sorted(set(whole['members'])-set(union['members'])),
                    actual_final_state_difference_squared=estimate(squared[0], squared[1:]),
                    final_state_nrmse_on_source_whole_scale=estimate(normalized[0], normalized[1:]),
                    primary_nrmse_union_minus_whole=estimate(primary_difference[0], primary_difference[1:]),
                    state_definition='Auxiliary actual ln_f state difference; primary full-vocabulary errors are reported separately for both calls')
                position_arrays[mechanism+'__'+method+'__union_minus_whole_ln_f_squared'] = state_squared
            print(json.dumps(dict(stage='MECHANISM_COMPLETE', mechanism=mechanism, **cohorts[mechanism])), flush=True)
    output.mkdir(parents=True)
    np.savez_compressed(output/'POSITION_METRICS.npz', document_bootstrap_multiplicity=multiplicity[1:],
                        bootstrap_document_order=np.array(all_documents), **position_arrays)
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        inputs=[run['input'] for run in loaded], analyzer=identity(Path(__file__)),
        statistics=summaries, paired_contrasts=contrasts, singletons=member_tables,
        union_vs_whole=union_checks, cohorts=cohorts,
        bootstrap=dict(draws=draws, seed=seed, documents=len(all_documents),
            unit='Global document clusters shared across all included methods, requests, members and mechanisms; repeated positions keep their document multiplicity',
            empty_strata='Mark the mechanism draw invalid and retain counts; no resampling until success'),
        definitions=dict(primary='sqrt(mean_stratum(mean_position(weighted_centered_logit_error_squared)) / mean_stratum(mean_position(own_source1_response_energy)))',
            vocabulary_weight='Actual clean softmax over complete vocabulary; subtract each effect clean-probability-weighted mean',
            requests='Every request and mechanism is reported independently; no pooled near-zero partial or complement denominator',
            singletons='Every source1 member and its source1 PW image use their own source1 response energy; zero energy gives undefined nRMSE',
            raw_positions='Original completed chunks retain ln_f states, token effects and masks; POSITION_METRICS retains all scalar observations and bootstrap identities'),
        cost=dict(cpu_wall_seconds=time.perf_counter()-start, model_forwards=0, new_fits=0))
    (output/'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (output/'SINGLETONS.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['mechanism', 'source_member', 'PW_target_member', 'nrmse',
                         'ci95_lower', 'ci95_upper', 'source_energy', 'error_squared', 'valid_bootstrap'])
        for mechanism, table in member_tables.items():
            for row in table['rows']:
                cell = row['statistics']
                interval = cell['nrmse']['ci95'] or [None, None]
                writer.writerow([mechanism, row['source_member'], row['target_member'],
                    cell['nrmse']['value'], *interval, cell['source_energy']['value'],
                    cell['error_squared']['value'], cell['nrmse']['valid_bootstrap']])
    lines = ['# Fixed operation responses', '',
        'Four token strata have equal weight. Each request uses its own actual source response scale. Document draws are shared; empty-stratum draws are excluded with their counts retained.', '',
        '| Mechanism | Request | Method | nRMSE [95% interval] | Source RMS | Next-token logp change |',
        '|---|---|---|---|---|---|']
    for mechanism in cohorts:
        for request in REQUESTS:
            for method in METHODS:
                cell = summaries[mechanism+'/'+method+'/'+request]
                value = cell['nrmse']
                text = 'undefined' if value['value'] is None else f"{value['value']:.6f}"
                if value['ci95'] is not None:
                    text += f" [{value['ci95'][0]:.6f}, {value['ci95'][1]:.6f}]"
                lines.append(f"| {mechanism} | {request} | {method} | {text} | {cell['source_rms']['value']:.6f} | {cell['next_token_logprob_change']['value']:.6f} |")
        cohort = cohorts[mechanism]
        lines.extend(['', f"{mechanism}: {cohort['positions']} positions in {cohort['documents']} documents; {cohort['valid_bootstrap']}/{draws} valid document draws.", ''])
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), mechanisms=list(cohorts), seconds=time.perf_counter()-start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260923)
    args = parser.parse_args()
    analyze(args.runs, args.output, args.bootstrap, args.seed)
