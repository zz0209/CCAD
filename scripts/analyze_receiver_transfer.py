import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


RECEIVERS = ['plural', 'singular', 'both']
ANSWER_NUMBER = {'are': 'plural', 'were': 'plural', 'have': 'plural',
                 'go': 'plural', 'do': 'plural', 'is': 'singular',
                 'was': 'singular', 'has': 'singular', 'goes': 'singular', 'does': 'singular'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def interval(values):
    finite = np.isfinite(values[1:])
    return dict(value=float(values[0]) if np.isfinite(values[0]) else None,
                ci95=np.quantile(values[1:][finite], [.025, .975]).tolist() if finite.any() else None,
                bootstrap_defined=int(finite.sum()))


def analyze(run, output, bootstrap):
    assert not output.exists(), output
    assert read(run/'status.json')['status'] == 'PASS'
    panel = read(run/'panel.json')
    rows = panel['rows']
    data = dict(np.load(run/'responses.npz', allow_pickle=False))
    names = data['operations'].tolist()
    margin = data['margin'].astype(np.float64)
    logits = data['answer_logits'].astype(np.float64)
    assert margin.shape == (len(names), len(rows))
    assert logits.shape == (*margin.shape, 2)
    assert np.isfinite(margin).all() and np.isfinite(logits).all()
    projection_error = float(np.max(np.abs(margin-(logits[:, :, 0]-logits[:, :, 1]))))
    m = {name: margin[i] for i, name in enumerate(names)}
    values = {'A_effect': m['A']-m['none']}
    for name in names:
        values[name+'_margin'] = m[name]
        values[name+'_correct'] = (m[name] > 0).astype(float)
    dz = data['b_codesA'].astype(float)-data['b_codes0'].astype(float)
    for j, receiver in enumerate(RECEIVERS):
        r = m['receiver_'+receiver]-m['none']
        restoration = m['restore_'+receiver]-m['A']
        values['receiver_'+receiver+'_effect'] = r
        values['restore_'+receiver+'_effect'] = restoration
        values['restoration_inverse_'+receiver] = -restoration
        values['finite_interaction_'+receiver] = -restoration-r
        if receiver != 'both':
            values['delta_code_'+receiver] = dz[:, j]
            values['code0_'+receiver] = data['b_codes0'][:, j].astype(float)
            values['codeA_'+receiver] = data['b_codesA'][:, j].astype(float)
            values['B_'+receiver+'_effect'] = m['B_'+receiver]-m['none']
            values['B_'+receiver+'_deletion_harm'] = m['none']-m['B_'+receiver]
    for kind in ['receiver', 'restoration_inverse']:
        suffix = '_effect' if kind == 'receiver' else ''
        values[kind+'_nonadditivity'] = (values[kind+'_both'+suffix]
            -values[kind+'_plural'+suffix]-values[kind+'_singular'+suffix])
    values['receiver_plural_minus_singular'] = (values['receiver_plural_effect']
                                                       -values['receiver_singular_effect'])
    values['natural_harm_plural_minus_singular'] = (values['B_plural_deletion_harm']
                                                       -values['B_singular_deletion_harm'])
    natural = np.column_stack([values['B_'+r+'_effect'] for r in RECEIVERS[:2]])
    receiver = np.column_stack([values['receiver_'+r+'_effect'] for r in RECEIVERS[:2]])
    # 强度排序与有符号作用共同保存，精确并列保持未定义。
    natural_difference = np.abs(natural[:, 0])-np.abs(natural[:, 1])
    induced_difference = np.abs(receiver[:, 0])-np.abs(receiver[:, 1])
    ranked = (natural_difference != 0) & (induced_difference != 0)
    values['absolute_rank_disagreement'] = np.where(ranked,
        (np.sign(natural_difference) != np.sign(induced_difference)).astype(float), np.nan)
    values['natural_rank_receiver_regret'] = np.where(natural_difference != 0,
        np.max(np.abs(receiver), axis=1)-np.abs(receiver)[np.arange(len(rows)),
                                                              (natural_difference < 0).astype(int)], np.nan)
    values['A_aligned_plural_minus_singular'] = np.where(values['A_effect'] != 0,
        np.sign(values['A_effect'])*values['receiver_plural_minus_singular'], np.nan)
    values['delta_hidden_norm'] = np.linalg.norm(data['hA']-data['h0'], axis=1)
    for j, name in enumerate(RECEIVERS[:2]):
        values['delta_'+name+'_norm'] = np.linalg.norm(data['b_delta'][:, j], axis=1)

    meta = []
    for i, row in enumerate(rows):
        answer = row.get('answer', row.get('recipient_answer'))
        number = row.get('correct_number', ANSWER_NUMBER[answer.strip()])
        pair = '|'.join(sorted([row['document_sha256'], row['donor_document_sha256']]))
        meta.append(dict(index=i, canonical_pair=pair, document_sha256=row['document_sha256'],
            structure=row['structure'], direction=row['direction'], correct_number=number,
            case=row.get('case', row['original_pair']['case'])))
    pairs = sorted(set(row['canonical_pair'] for row in meta))
    pair_index = {pair: i for i, pair in enumerate(pairs)}
    # 同一方向的重复输入在每个分析分组内仅计一次。
    groups = {'all': np.arange(len(rows))}
    for field in ['structure', 'direction', 'correct_number', 'case']:
        for value in sorted(set(row[field] for row in meta)):
            groups[field+'='+str(value)] = np.array([r['index'] for r in meta if r[field] == value])
    for structure in sorted(set(row['structure'] for row in meta)):
        for number in ['plural', 'singular']:
            groups['structure='+structure+'/number='+number] = np.array([
                r['index'] for r in meta if r['structure'] == structure and r['correct_number'] == number])
    for structure, case, direction in sorted(set((r['structure'], r['case'], r['direction']) for r in meta)):
        groups[f'{structure}/{case}/{direction}'] = np.array([r['index'] for r in meta
            if (r['structure'], r['case'], r['direction']) == (structure, case, direction)])
    strata = {}
    for p, pi in pair_index.items():
        structures = {r['structure'] for r in meta if r['canonical_pair'] == p}
        assert len(structures) == 1
        strata.setdefault(next(iter(structures)), []).append(pi)
    rng = np.random.default_rng(9231413)
    counts = np.ones((bootstrap+1, len(pairs)), dtype=float)
    for k in range(1, bootstrap+1):
        counts[k] = 0
        for indices in strata.values():
            sample = rng.choice(indices, size=len(indices), replace=True)
            counts[k] += np.bincount(sample, minlength=len(pairs))

    result_groups = {}
    for group_name, indices in groups.items():
        if not len(indices):
            continue
        unique = {}
        for i in indices:
            unique.setdefault((meta[i]['canonical_pair'], meta[i]['document_sha256']), []).append(i)
        units = list(unique.values())
        pindices = np.array([pair_index[meta[unit[0]]['canonical_pair']] for unit in units])
        # 先平均同一输入的重复计算，再使每个句对内输入方向等权。
        multiplicity = np.bincount(pindices, minlength=len(pairs))
        weights = counts[:, pindices]/multiplicity[pindices][None, :]
        stats = {}
        for metric, vector in values.items():
            unit_values = np.array([np.mean(vector[unit]) for unit in units])
            valid = np.isfinite(unit_values)
            numerator = weights[:, valid]@unit_values[valid]
            denominator = weights[:, valid].sum(axis=1)
            means = np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)
            stats[metric] = dict(interval(means), defined_inputs=int(valid.sum()))
            if metric.endswith('_effect') or metric.startswith('finite_interaction_'):
                moments = np.divide(weights[:, valid]@(unit_values[valid]**2), denominator,
                    out=np.full(len(weights), np.nan), where=denominator > 0)
                stats[metric+'_rms'] = interval(np.sqrt(moments))
        result_groups[group_name] = dict(rows=len(indices), unique_inputs=len(units),
            canonical_pairs=len(set(pindices)), metrics=stats)

    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        inputs={name: identity(run/name) for name in ['panel.json', 'responses.npz', 'status.json']},
        analyzer=identity(Path(__file__)), operations=names, row_count=len(rows),
        canonical_pair_count=len(pairs), unique_documents=len(set(r['document_sha256'] for r in rows)),
        dependence=dict(unit='canonical unordered text pair', strata={k:len(v) for k,v in strata.items()},
            repetitions=bootstrap, seed=9231413, pair_weight='equal within each reported group',
            structure_weight='proportional to canonical pair counts',
            duplicates='Repeated text directions averaged before pair-level weighting',
            scope='Conditional on the fixed source SAE, named nodes and exposed development structures'),
        definitions=dict(receiver_effect='margin(receiver)-margin(none)',
            restoration_inverse='margin(A)-margin(restore)',
            finite_interaction='restoration_inverse-receiver_effect',
            deletion_harm='margin(none)-margin(B)',
            ranking='Absolute operation strengths; signed effects separately retained; exact ties undefined',
            A_aligned_difference='sign(A effect)*(receiver_plural effect-receiver_singular effect)'),
        checks=dict(margin_from_logits_max_error=projection_error), groups=result_groups,
        rows=[dict(metadata=meta[i], original=rows[i], metrics={key: float(v[i]) if np.isfinite(v[i]) else None
            for key,v in values.items()}) for i in range(len(rows))])
    output.mkdir(parents=True)
    (output/'RESULTS.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    np.savez_compressed(output/'row_statistics.npz', **values)
    lines = ['# Source receiver 作用测量', '',
        f'{len(rows)} 个原始输入方向，{len(pairs)} 个 canonical pair。按语法结构内句对共同 bootstrap {bootstrap} 次。',
        '每个 margin 正方向均为当前输入正确答案；同一输入的重复记录在统计中共同计入一次。', '',
        '| 条件 | 句对 | A | R plural | R singular | M plural | M singular | 交互 plural | 交互 singular |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    columns = ['A_effect', 'receiver_plural_effect', 'receiver_singular_effect',
               'restoration_inverse_plural', 'restoration_inverse_singular',
               'finite_interaction_plural', 'finite_interaction_singular']
    for name, group in result_groups.items():
        cells = [f'{group["metrics"][key]["value"]:.6f}' for key in columns]
        lines.append('| '+name+' | '+str(group['canonical_pairs'])+' | '+' | '.join(cells)+' |')
    lines += ['', '完整有符号作用、Δcode、自然删除排序、receiver 强度排序及逐输入文本保存在 RESULTS.json。',
              '排序差异只说明当前源节点的作用区别；跨目标预测能力由后续冻结关系的实际调用回答。']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), rows=len(rows), pairs=len(pairs),
                         all=result_groups['all']['metrics']), ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--check-existing', action='store_true')
    parser.add_argument('--stable-ties', action='store_true')
    args = parser.parse_args()
    assert args.bootstrap > 0
    if args.check_existing:
        check_existing(args.run, args.output, args.stable_ties)
    else:
        analyze(args.run, args.output, args.bootstrap)


def check_existing(run, output, stable_ties=False):
    destination = output/('RAW_STABLE_TIE_CHECK.json' if stable_ties else 'RAW_CHECK.json')
    assert not destination.exists()
    saved = read(output/'RESULTS.json')
    rows = read(run/'panel.json')['rows']
    raw = dict(np.load(run/'responses.npz', allow_pickle=False))
    positions = {name:i for i,name in enumerate(raw['operations'].tolist())}
    logits = raw['answer_logits'].astype(float)
    margins = logits[..., 0]-logits[..., 1]
    unique = {}
    for i,row in enumerate(rows):
        unique.setdefault(row['document_sha256'], []).append(i)
    indices = list(unique.values())
    f = np.column_stack([margins[:, ii].mean(axis=1) for ii in indices])
    a = f[positions['A']]-f[positions['none']]
    direct = {}
    receiver, natural = [], []
    for name in ['plural', 'singular']:
        r = f[positions['receiver_'+name]]-f[positions['none']]
        s = f[positions['A']]-f[positions['restore_'+name]]
        b = f[positions['B_'+name]]-f[positions['none']]
        receiver.append(r)
        natural.append(b)
        direct['receiver_'+name+'_effect'] = r.mean()
        direct['restoration_inverse_'+name] = s.mean()
        direct['finite_interaction_'+name] = (s-r).mean()
        direct['B_'+name+'_effect'] = b.mean()
    direct['A_effect'] = a.mean()
    direct['receiver_nonadditivity'] = (f[positions['receiver_both']]-f[positions['none']]
                                                      -np.array(receiver).sum(axis=0)).mean()
    errors = {name: float(abs(value-saved['groups']['all']['metrics'][name]['value']))
              for name,value in direct.items()}
    assert max(errors.values()) < 1e-7
    receiver, natural = np.array(receiver).T, np.array(natural).T
    nstrong, rstrong = np.abs(natural), np.abs(receiver)
    nmax, rmax = nstrong.max(axis=1), rstrong.max(axis=1)
    choices = nstrong == nmax[:, None]
    natural_ties = int((choices.sum(axis=1)==2).sum())
    best = rstrong == rmax[:, None]
    if stable_ties:
        choices = np.arange(2)[None, :] == np.argmax(nstrong, axis=1)[:, None]
    # 实际调用采用成员 ID 3982 优先；receiver 并列均为正确选择。
    success = (choices*best).sum(axis=1)/choices.sum(axis=1)
    chosen_strength = (choices*rstrong).sum(axis=1)/choices.sum(axis=1)
    ranking = dict(inputs=len(indices), natural_ties=natural_ties,
        receiver_ties=int((best.sum(axis=1)==2).sum()),
        both_strict=int(((nstrong[:,0]!=nstrong[:,1])&(best.sum(axis=1)==1)).sum()),
        tie_rule='smaller member ID 3982' if stable_ties else 'equal probability among normal-strength ties',
        zero_normal_nonzero_receiver=int(((nmax==0)&(rmax>0)).sum()),
        natural_choice_success=float(success.mean()),
        mean_regret=float((rmax-chosen_strength).mean()), max_regret=float((rmax-chosen_strength).max()),
        disagreements=[dict(text=rows[ii[0]]['text'], answer=rows[ii[0]]['answer'],
            natural=natural[j].tolist(), receiver=receiver[j].tolist(), regret=float(rmax[j]-chosen_strength[j]))
            for j,ii in enumerate(indices) if success[j] < 1])
    result = dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        inputs=saved['inputs'], independent_mean_errors=errors,
        scope='Direct answer-logit subtraction and unique-text averaging; paired directions are balanced in this source panel',
        ranking_all_inputs=ranking, exact_zero_A_effect_inputs=int((a==0).sum()),
        first_row=dict(text=rows[0]['text'], answer=rows[0]['answer'],
            answer_logits=logits[:,0].tolist(), operations=raw['operations'].tolist(),
            delta_code=(raw['b_codesA'][0]-raw['b_codes0'][0]).tolist()))
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
