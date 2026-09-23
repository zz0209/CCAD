from pathlib import Path
import argparse
import json
import platform
import sys
import time
import traceback

import numpy as np

from run_causalgym_multisite import MultisiteWork, write
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor
from ccad.artifacts import sha256


FACTORS = ['number', 'time']
OPERATIONS = [('number', 'number'), ('time', 'number'), ('number', 'time'),
              ('time', 'time'), ('number', 'joint'), ('time', 'joint'), ('joint', 'joint')]


def load_config(path):
    current = json.loads(Path(path).read_text())
    result = load_config(current['base_config']) if current.get('base_config') else {}
    result.update(current)
    return result


def read_panel(work, path):
    panel = json.loads(work.checked(path, 'Explicit grammar panel').read_text())
    rows, pairs = panel['rows'], panel['pairs']
    assert len(rows) == len(pairs) and len(rows) > 0
    for i, (row, pair) in enumerate(zip(rows, pairs)):
        assert pair['base'] == i
        for factor in FACTORS+['joint']:
            donor = rows[pair[factor]]
            for axis in FACTORS:
                expected = 1-int(row[axis]) if factor in [axis, 'joint'] else int(row[axis])
                assert int(donor[axis]) == expected
            for name in ['attractor', 'subject_id', 'attractor_id', 'cue_id', 'template']:
                assert donor[name] == row[name]
    return panel


def dense_codes(sae, hidden, torch, device, batch_size):
    result = np.empty((len(hidden), sae.num_latents), np.float32)
    with torch.no_grad():
        for start in range(0, len(hidden), batch_size):
            x = torch.as_tensor(hidden[start:start+batch_size], device=device)
            values, indices, _ = sae.encode(x)
            codes = torch.zeros((len(x), sae.num_latents), device=device).scatter_(1, indices, values)
            result[start:start+len(x)] = codes.cpu().numpy()
    assert np.isfinite(result).all()
    return result


def source_updates(codes, decoder, members, pairs):
    return {factor: (codes[[pair[factor] for pair in pairs]][:, members[factor]]
                    - codes[:, members[factor]]) @ decoder[members[factor]] for factor in FACTORS}


def fit_fraction(delta_codes, decoder, teacher, torch, device):
    x = torch.as_tensor(delta_codes, device=device)
    d = torch.as_tensor(decoder, device=device)
    y = torch.as_tensor(teacher, device=device)
    scale = y.square().mean()
    assert torch.isfinite(scale) and scale > 0
    weight = torch.nn.Parameter(torch.zeros(x.shape[1], device=device))
    optimizer = torch.optim.Adam([weight], lr=.03)
    trace = []
    for step in range(240):
        optimizer.zero_grad()
        prediction = (x*weight)@d
        loss = (prediction-y).square().mean()/scale
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            weight.clamp_(0, 1)
        trace.append(float(loss.detach()))
    with torch.no_grad():
        final_loss = float((((x*weight)@d-y).square().mean()/scale))
    return weight.detach().cpu().numpy(), dict(raw_scale=float(scale), loss_trace=trace,
        final_loss=final_loss, steps=240, learning_rate=.03, initial='zero')


def fit_raw(x, teacher, source_decoder):
    # teacher 位于固定源 decoder 的物理空间，低维输出保留其实际 rank。
    _, singular, basis = np.linalg.svd(source_decoder.astype(np.float64), full_matrices=False)
    tolerance = max(source_decoder.shape)*np.finfo(np.float64).eps*singular[0]
    physical_rank = int((singular > tolerance).sum())
    basis = basis[:physical_rank]
    x = x.astype(np.float64)
    coordinates = teacher.astype(np.float64)@basis.T
    input_rank = int(np.linalg.matrix_rank(x))
    assert input_rank > 0 and physical_rank > 0
    ridge = 1e-3*float(np.square(x).sum())/input_rank
    dual = np.linalg.solve(x@x.T+ridge*np.eye(len(x)), coordinates)
    left = x.T@dual
    return left.astype(np.float32), basis.astype(np.float32), dict(ridge=ridge,
        input_rank=input_rank, source_physical_rank=physical_rank,
        fitted_output_rank=int(np.linalg.matrix_rank(left)), samples=len(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault('generator_script', 'scripts/evaluate_grammar_material_support.py')
    mode = cfg.get('mode', 'fit_evaluate')
    assert mode in ['fit', 'fit_evaluate', 'evaluate']
    work = MultisiteWork(cfg, args.config, ['scripts/evaluate_grammar_material_support.py',
        'scripts/run_causalgym_multisite.py', 'src/ccad/activation_contract.py', 'src/ccad/artifacts.py'])
    error = None
    fitting_backwards = encoded_states = tail_forwards = 0
    try:
        import torch
        import transformers
        sys.path.insert(0, cfg['sparsify_overlay_dir'])
        sys.path.insert(0, cfg['sparsify_source_dir'])
        from sparsify import SparseCoder
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('high')
        work.torch, work.device = torch, torch.device(cfg['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
            dtype='float32', matmul_precision='high')
        for filename in ['sparse_coder.py', 'fused_encoder.py']:
            work.checked(Path(cfg['sparsify_source_dir'])/'sparsify'/filename, 'Frozen Sparsify source', 'MIT')
        model_path = Path(cfg['model_local_dir'])
        for filename in ['config.json', 'model.safetensors', 'tokenizer.json']:
            work.checked(model_path/filename, 'Frozen base model', 'Apache-2.0')
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').to(work.device).eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        assert model.config.num_hidden_layers == 16 and model.config.hidden_size == 2048
        label_words = cfg.get('label_words', [' is', ' are', ' was', ' were'])
        label_ids = [tokenizer.encode(word, add_special_tokens=False) for word in label_words]
        assert all(len(ids) == 1 for ids in label_ids)
        label_ids = [ids[0] for ids in label_ids]
        output_weight = model.get_output_embeddings().weight[label_ids].detach()
        module = model.gpt_neox.layers[15]
        contract = HookPointContract('gpt_neox.layers.15', 15, 'resid_post', 2048)
        batch_size = int(cfg.get('batch_size', 32))

        def tail(hidden):
            nonlocal tail_forwards
            result = []
            with torch.no_grad():
                for start in range(0, len(hidden), batch_size):
                    x = torch.as_tensor(hidden[start:start+batch_size], device=work.device)
                    result.append((model.gpt_neox.final_layer_norm(x)@output_weight.T).cpu().numpy())
            tail_forwards += len(hidden)
            return np.concatenate(result)

        def forward(rows, delta=None):
            if time.perf_counter()-work.wall_start > cfg['budget_seconds']:
                raise TimeoutError('Grammar functional evaluation budget exceeded')
            enc = tokenizer([row['text'] for row in rows], add_special_tokens=False,
                            padding=True, return_tensors='pt').to(work.device)
            last = enc.attention_mask.sum(1)-1
            ix = torch.arange(len(rows), device=work.device)
            captured = []
            def hook(_module, _args, output):
                hidden = extract_primary_hook_tensor(output, contract)
                captured.append(hidden[ix, last].detach().cpu().numpy())
                if delta is not None:
                    hidden = hidden.clone()
                    hidden[ix, last] += torch.as_tensor(delta, device=work.device)
                    return replace_primary_hook_tensor(output, hidden, contract)
            handle = module.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    hidden = model.gpt_neox(**enc, use_cache=False).last_hidden_state[ix, last]
                    logits = (hidden@output_weight.T).cpu().numpy()
            finally:
                handle.remove()
            work.sequence_forwards += len(rows)
            work.token_forwards += int(enc.attention_mask.sum())
            assert len(captured) == 1
            return captured[0], logits

        def capture(panel, name):
            states, baseline = [], []
            for start in range(0, len(panel['rows']), batch_size):
                h, values = forward(panel['rows'][start:start+batch_size])
                states.append(h)
                baseline.append(values)
            states, baseline = np.concatenate(states), np.concatenate(baseline)
            error_value = float(np.max(np.abs(tail(states)-baseline)))
            assert error_value <= 1e-4, error_value
            work.checks[name+'_current_tail_baseline'] = True
            np.savez_compressed(work.run/(name+'_current_states.npz'), hidden=states,
                baseline_logits4=baseline, label_ids=np.asarray(label_ids))
            work.progress('CURRENT_STATES_CAPTURED', panel=name, rows=len(states), tail_error=error_value)
            return states, baseline

        def load_sae(path):
            path = Path(path)
            work.checked(path/'cfg.json', 'Frozen SAE configuration')
            weights = work.checked(path/'sae.safetensors', 'Frozen SAE weights')
            sae = SparseCoder.load_from_disk(path, device=str(work.device)).float().eval()
            sae.requires_grad_(False)
            assert sae.d_in == 2048 and sae.num_latents == 8192 and sae.cfg.k == 64
            return sae, sha256(weights)

        source, source_hash = load_sae(cfg['source_checkpoint'])
        source_decoder = source.W_dec.detach().cpu().numpy()
        component_path = work.checked(cfg['source_components'], 'Original source1 N16/T32 members')
        components = json.loads(component_path.read_text())['rows']
        members = {factor: np.array(next(row['members'] for row in components
                    if row['seed'] == 1 and row['factor'] == factor and row['budget'] == budget), dtype=np.int64)
                   for factor, budget in [('number', 16), ('time', 32)]}
        relation_dir = work.run/'relations'
        relation_dir.mkdir()
        target_specs = cfg['target_checkpoints']
        assert len({(spec['seed'], spec['arm']) for spec in target_specs}) == len(target_specs)
        relation_index = dict(source_checkpoint_sha256=source_hash,
            source_components_sha256=sha256(component_path), targets=[], source_members={
                key: value.tolist() for key, value in members.items()}, label_ids=label_ids)
        relations = {}
        raw_maps = {}
        if mode in ['fit', 'fit_evaluate']:
            full_cal = read_panel(work, cfg['calibration_panel'])
            selected = [i for i, row in enumerate(full_cal['rows']) if row['cue_id'] == 0]
            assert len(selected) == 64
            inverse = {old: new for new, old in enumerate(selected)}
            cal = dict(rows=[full_cal['rows'][i] for i in selected], pairs=[
                {key: inverse[value] for key, value in full_cal['pairs'][i].items()} for i in selected])
            write(work.run/'calibration_membership.json', dict(original_indices=selected, **cal))
            hidden, _ = capture(cal, 'calibration')
            source_codes = dense_codes(source, hidden, torch, work.device, batch_size)
            encoded_states += len(hidden)
            updates = source_updates(source_codes, source_decoder, members, cal['pairs'])
            donor_ids = {factor: np.array([pair[factor] for pair in cal['pairs']]) for factor in FACTORS}
            x_raw = np.concatenate([hidden[donor_ids[factor]]-hidden for factor in FACTORS])
            teachers = {factor: np.concatenate([updates[factor] if axis == factor else np.zeros_like(hidden)
                                               for axis in FACTORS]) for factor in FACTORS}
            raw_saved, raw_diagnostics = {}, {}
            for factor in FACTORS:
                left, basis, diagnostics = fit_raw(x_raw, teachers[factor], source_decoder[members[factor]])
                raw_maps[factor] = (left, basis)
                raw_saved[factor+'__left'], raw_saved[factor+'__basis'] = left, basis
                raw_diagnostics[factor] = diagnostics
            np.savez_compressed(relation_dir/'raw.npz', **raw_saved)
            write(relation_dir/'raw_fit.json', raw_diagnostics)
            np.savez_compressed(relation_dir/'calibration_information.npz', hidden=hidden,
                source_codes=source_codes, raw_inputs=x_raw,
                number_teacher=teachers['number'], time_teacher=teachers['time'])
            relation_index.update(calibration_panel_sha256=sha256(Path(cfg['calibration_panel'])),
                calibration_indices=selected, raw_path=str(relation_dir/'raw.npz'),
                raw_sha256=sha256(relation_dir/'raw.npz'))
            for spec in target_specs:
                target, target_hash = load_sae(spec['path'])
                decoder = target.W_dec.detach().cpu().numpy()
                codes = dense_codes(target, hidden, torch, work.device, batch_size)
                encoded_states += len(hidden)
                mask = np.zeros((8192, 2), np.float32)
                saved, diagnostics = {}, {}
                unit = decoder/np.maximum(np.linalg.norm(decoder, axis=1, keepdims=True), 1e-30)
                for column, factor in enumerate(FACTORS):
                    source_d = source_decoder[members[factor]]
                    source_unit = source_d/np.maximum(np.linalg.norm(source_d, axis=1, keepdims=True), 1e-30)
                    score = np.maximum(unit@source_unit.T, 0).max(1)
                    candidates = np.argsort(-score, kind='stable')[:256]
                    delta_codes = np.concatenate([codes[donor_ids[axis]][:, candidates]-codes[:, candidates]
                                                  for axis in FACTORS])
                    weights, diag = fit_fraction(delta_codes, decoder[candidates], teachers[factor], torch, work.device)
                    fitting_backwards += 240
                    mask[candidates, column] = weights
                    saved.update({factor+'__candidates': candidates, factor+'__weights': weights,
                                  factor+'__delta_codes': delta_codes, factor+'__cosine_scores': score[candidates]})
                    diagnostics[factor] = diag
                key = f"{spec['arm']}_s{spec['seed']}"
                path = relation_dir/(key+'.npz')
                np.savez_compressed(path, mask=mask, **saved)
                write(relation_dir/(key+'_fit.json'), diagnostics)
                item = dict(**spec, relation_path=str(path), checkpoint_sha256=target_hash,
                            relation_sha256=sha256(path), key=key)
                relation_index['targets'].append(item)
                relations[key] = mask
                work.record(kind='fit', task='calibration', component=key, row_id=0, mode='fit',
                    method=spec['arm'], target_seed=int(spec['seed']), number_loss=diagnostics['number']['final_loss'],
                    time_loss=diagnostics['time']['final_loss'], calibration_rows=64)
                write(work.run/'relation_index.json', relation_index)
                work.progress('RELATION_FIT_COMPLETE', target_seed=spec['seed'], arm=spec['arm'])
                del target
        else:
            frozen = Path(cfg['frozen_relation_run'])
            relation_index = json.loads(work.checked(frozen/'relation_index.json', 'Frozen functional relation').read_text())
            assert relation_index['source_checkpoint_sha256'] == source_hash
            assert relation_index['source_components_sha256'] == sha256(component_path)
            assert relation_index['label_ids'] == label_ids
            assert {(x['seed'], x['arm']) for x in target_specs} == {(x['seed'], x['arm']) for x in relation_index['targets']}
            raw_path = work.checked(relation_index['raw_path'])
            assert sha256(raw_path) == relation_index['raw_sha256']
            raw = np.load(raw_path, allow_pickle=False)
            raw_maps = {factor: (raw[factor+'__left'], raw[factor+'__basis']) for factor in FACTORS}
            for spec in relation_index['targets']:
                path = work.checked(spec['relation_path'], 'Frozen target fractional masks')
                assert sha256(path) == spec['relation_sha256']
                relations[spec['key']] = np.load(path, allow_pickle=False)['mask']
            write(work.run/'relation_index.json', relation_index)
        evaluation_index = {}
        if mode != 'fit':
            assert cfg.get('evaluation_panels')
            for panel_name, panel_path in cfg['evaluation_panels'].items():
                panel = read_panel(work, panel_path)
                hidden, baseline = capture(panel, panel_name)
                rows, pairs = panel['rows'], panel['pairs']
                n = len(rows)
                source_codes = dense_codes(source, hidden, torch, work.device, batch_size)
                encoded_states += n
                updates = source_updates(source_codes, source_decoder, members, pairs)
                donors = {factor: np.array([pair[factor] for pair in pairs]) for factor in FACTORS+['joint']}
                source_mask = np.zeros((8192, 2), np.float32)
                for column, factor in enumerate(FACTORS):
                    source_mask[members[factor], column] = 1
                method_names = ['source_teacher', 'source_mask', 'source_full_sae', 'raw']+[s['key'] for s in relation_index['targets']]
                method_seeds = [1, 1, 1, 0]+[int(s['seed']) for s in relation_index['targets']]
                logits = np.empty((n, len(method_names), len(OPERATIONS), 4), np.float32)
                delta_norm = np.empty((n, len(method_names), len(OPERATIONS)), np.float32)
                expected = np.empty((n, len(OPERATIONS)), np.int64)
                full_check = {}

                def decode(values, decoder):
                    with torch.no_grad():
                        return (torch.as_tensor(values, device=work.device)
                                @ torch.as_tensor(decoder, device=work.device)).cpu().numpy()

                def evaluate_method(method_index, deltas):
                    name = method_names[method_index]
                    for operation_index, delta in enumerate(deltas):
                        logits[:, method_index, operation_index] = tail(hidden+delta)
                        delta_norm[:, method_index, operation_index] = np.linalg.norm(delta, axis=1)
                    # 当前批次的真实 hook 写回与缓存尾部使用同一批次形状。
                    chosen = 4
                    count = min(batch_size, n)
                    _, actual = forward(rows[:count], deltas[chosen][:count])
                    error_value = float(np.max(np.abs(actual-logits[:count, method_index, chosen])))
                    assert error_value <= 1e-4, (name, error_value)
                    full_check[name] = error_value

                teacher_deltas, mask_deltas, full_deltas, raw_deltas = [], [], [], []
                full_source = decode(source_codes, source_decoder)
                for oi, (request, donor_type) in enumerate(OPERATIONS):
                    axes = FACTORS if request == 'joint' else [request]
                    active_axes = [axis for axis in axes if donor_type in [axis, 'joint']]
                    teacher = sum((updates[axis] for axis in active_axes), start=np.zeros_like(hidden))
                    teacher_deltas.append(teacher)
                    number = np.array([row['number'] for row in rows])
                    past = np.array([row['time'] for row in rows])
                    if 'number' in active_axes:
                        number = 1-number
                    if 'time' in active_axes:
                        past = 1-past
                    expected[:, oi] = number+2*past
                    columns = [FACTORS.index(axis) for axis in axes]
                    weights = np.minimum(1, source_mask[:, columns].sum(1))
                    support = np.flatnonzero(weights)
                    dz = source_codes[donors[donor_type]][:, support]-source_codes[:, support]
                    mask_deltas.append(decode(dz*weights[support], source_decoder[support]))
                    full_deltas.append(full_source[donors[donor_type]]-full_source)
                    raw_x = hidden[donors[donor_type]]-hidden
                    raw_deltas.append(sum(((raw_x@raw_maps[axis][0])@raw_maps[axis][1] for axis in axes),
                                          start=np.zeros_like(hidden)))
                for mi, deltas in enumerate([teacher_deltas, mask_deltas, full_deltas, raw_deltas]):
                    evaluate_method(mi, deltas)
                for mi, spec in enumerate(relation_index['targets'], start=4):
                    target, target_hash = load_sae(spec['path'])
                    assert target_hash == spec['checkpoint_sha256']
                    codes = dense_codes(target, hidden, torch, work.device, batch_size)
                    encoded_states += n
                    decoder = target.W_dec.detach().cpu().numpy()
                    mask = relations[spec['key']]
                    deltas = []
                    for request, donor_type in OPERATIONS:
                        columns = [0, 1] if request == 'joint' else [FACTORS.index(request)]
                        weights = np.minimum(1, mask[:, columns].sum(1))
                        support = np.flatnonzero(weights)
                        dz = codes[donors[donor_type]][:, support]-codes[:, support]
                        deltas.append(decode(dz*weights[support], decoder[support]))
                    evaluate_method(mi, deltas)
                    np.savez_compressed(work.run/(panel_name+'__'+spec['key']+'__codes.npz'),
                        support=np.flatnonzero(mask.sum(1)), codes=codes[:, mask.sum(1) > 0])
                    del target
                centered = logits-logits.mean(-1, keepdims=True)
                effect = centered-(baseline-baseline.mean(-1, keepdims=True))[:, None, None]
                operation_names = [request+'_from_'+donor for request, donor in OPERATIONS]
                output_path = work.run/(panel_name+'_responses.npz')
                np.savez_compressed(output_path, logits4=logits, centered_effect=effect,
                    baseline_logits4=baseline, hidden_delta_norm=delta_norm, expected_label=expected,
                    method_names=np.asarray(method_names), method_seeds=np.asarray(method_seeds),
                    operation_names=np.asarray(operation_names), label_ids=np.asarray(label_ids),
                    row_ids=np.arange(n), source_teacher_delta=np.asarray(teacher_deltas).transpose(1, 0, 2))
                write(work.run/(panel_name+'_membership.json'), panel)
                write(work.run/(panel_name+'_full_model_checks.json'), full_check)
                for mi, name in enumerate(method_names):
                    for oi, operation in enumerate(operation_names):
                        records = []
                        for ri, row in enumerate(rows):
                            values = logits[ri, mi, oi]
                            probability = np.exp(values-values.max())
                            probability /= probability.sum()
                            number_hit = bool((probability[[1, 3]].sum() > .5) == (expected[ri, oi] % 2))
                            time_hit = bool((probability[[2, 3]].sum() > .5) == (expected[ri, oi] // 2))
                            records.append(dict(run_id=cfg['run_id'], metric_version='multisite-v1',
                                kind='intervention', task=panel_name, component=str(row['subject_id']),
                                row_id=ri, mode='finite_donor', method=name, target_seed=method_seeds[mi],
                                operation=operation, subject_id=row['subject_id'], cue_id=row['cue_id'],
                                template=row['template'], four_word_correct=bool(values.argmax() == expected[ri, oi]),
                                number_correct=number_hit, time_correct=time_hit,
                                joint_success=number_hit and time_hit, delta_norm=float(delta_norm[ri, mi, oi])))
                        with (work.run/'metrics.raw.jsonl').open('a', encoding='utf-8') as stream:
                            stream.writelines(json.dumps(record)+'\n' for record in records)
                        work.metrics.extend(records)
                    work.progress('METHOD_RECORDS_SAVED', panel=panel_name, method=name,
                        methods_completed=mi+1, methods_total=len(method_names),
                        records_saved=(mi+1)*len(operation_names)*n)
                evaluation_index[panel_name] = dict(response_path=str(output_path),
                    membership_path=str(work.run/(panel_name+'_membership.json')),
                    panel_sha256=sha256(Path(panel_path)), rows=n)
                write(work.run/'evaluation_index.json', evaluation_index)
                work.checks[panel_name+'_current_full_intervention'] = True
                work.progress('PANEL_COMPLETE', panel=panel_name, rows=n, methods=len(method_names))
        work.checks['relations_complete'] = len(relations) == len(target_specs)
    except Exception:
        error = traceback.format_exc()
    write(work.run/'execution_cost.json', dict(fitting_backward_calls=fitting_backwards,
        encoded_states=encoded_states, cached_tail_sequence_forwards=tail_forwards,
        full_model_sequence_forwards=work.sequence_forwards, full_model_token_forwards=work.token_forwards))
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
