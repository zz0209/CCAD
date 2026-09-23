from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback

os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np
import torch
import transformers

from ccad.artifacts import sha256
from run_causalgym_multisite import MultisiteWork, write
from functional_fragment_model import fit_basis, save_basis, load_basis, carrier_masks, operation_delta, role_projection, whole_energy_carriers, match_source_carriers, calibration_scales, singleton_oracle


OPERATIONS = ('clean', 'F', 'L', 'F+L', 'F_carrier4_fragment', 'L_carrier4_fragment', 'F_carrier4_whole', 'L_carrier4_whole')


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class FragmentRunner:
    def __init__(self, work, config):
        self.work, self.config = work, config
        torch.set_num_threads(config.get('cpu_threads', 4))
        torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True)
        work.torch, work.device = torch, torch.device(config['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, python_version=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__, matmul_precision='highest', dtype='float32', gpu=torch.cuda.get_device_name(work.device))
        panel_path = work.checked(config['panel'])
        self.panel = json.loads(panel_path.read_text())
        self.answers = torch.tensor(self.panel['answer_ids'], device=work.device)
        self.rows = []
        splits = ('mean', 'fit', 'calibration', 'development') if config['phase'] == 'source' else config.get('evaluation_splits', ['calibration', 'development'])
        for split in splits:
            selected = [row for row in self.panel['rows'] if row['split'] == split]
            limit = config.get('split_limits', {}).get(split, len(selected))
            self.rows.extend(selected[:limit])
        assert config['phase'] != 'source' or all(row['split'] != 'confirmation' for row in self.rows)
        write(work.run/'panel.json', dict(rows=self.rows, icl_families=self.panel['icl_families'], answer_ids=self.panel['answer_ids'], panel_path=str(panel_path), panel_sha256=sha256(panel_path)))
        root = Path(config['model_local_dir'])
        for name in ('config.json', 'model.safetensors', 'tokenizer.json'):
            work.checked(root/name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(root, local_files_only=True, dtype=torch.float32, attn_implementation='eager').to(work.device).eval().requires_grad_(False)
        self.model.config.use_cache = False
        sys.path.extend([config['dictionary_source_dir'], config['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        self.saes, checkpoints = {}, {}
        seeds = config.get('mean_seeds', [1, 2, 3, 4, 5]) if config['phase'] == 'source' else sorted({1, config['target_seed']})
        for seed in seeds:
            path = Path(config['checkpoint_directory'])/f'topk_seed{seed}.pt'
            if seed == 1 and config.get('source_checkpoint'):
                path = Path(config['source_checkpoint'])
            work.checked(path)
            state = torch.load(path, map_location=work.device, weights_only=True)
            sae = AutoEncoderTopK(self.model.config.hidden_size, state['encoder.weight'].shape[0], int(state['k'])).to(work.device)
            sae.load_state_dict(state)
            self.saes[seed] = sae.eval().requires_grad_(False)
            checkpoints[str(seed)] = dict(path=str(path), sha256=sha256(path))
        assert 1 in self.saes
        self.decoder = self.saes[1].decoder.weight.T.detach().cpu().numpy()
        self.source_bias = self.saes[1].b_dec.detach().cpu().numpy()
        self.positions, self.delta, self.captured = None, None, None
        self.handle = self.model.get_submodule(config.get('hook_module_path', 'model.layers.13')).register_forward_hook(self.hook)
        self.identity = dict(panel_sha256=sha256(panel_path), model_revision=config['model_revision'], checkpoint_identities=checkpoints, hook=config.get('hook_module_path', 'model.layers.13'), split_limits=config.get('split_limits', {}), batch_size=config['batch_size'], phase=config['phase'], evaluation_splits=list(splits), matmul_precision='highest', operations=list(OPERATIONS), code_sha256=sha256(work.run/'source_snapshot/scripts/run_functional_fragments.py'), model_code_sha256=sha256(work.run/'source_snapshot/scripts/functional_fragment_model.py'), adaptation='Qwen tokenizer words; first/last-letter source functional fragments')
        self.identity_string = identity_hash(self.identity)
        write(work.run/'identity.json', self.identity)
        self.previous = None
        if config.get('resume_run'):
            previous = Path(config['resume_run'])
            assert json.loads((previous/'identity.json').read_text()) == self.identity
            self.previous = previous

    def hook(self, module, inputs, output):
        values = output[0] if isinstance(output, tuple) else output
        index = torch.arange(len(values), device=self.work.device)
        self.captured = values[index, self.positions].detach().clone()
        if self.delta is None:
            return output
        changed = values.clone()
        changed[index, self.positions] += self.delta
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    def forward(self, rows, delta=None):
        prompts = [row['prompts'][task] for row in rows for task in ('first', 'last')]
        length = max(len(row['tokens']) for row in prompts)
        ids = torch.full((len(prompts), length), self.tokenizer.eos_token_id, device=self.work.device, dtype=torch.long)
        attention = torch.zeros_like(ids)
        for i, row in enumerate(prompts):
            ids[i, :len(row['tokens'])] = torch.tensor(row['tokens'], device=self.work.device)
            attention[i, :len(row['tokens'])] = 1
        self.positions = torch.tensor([row['word_position'] for row in prompts], device=self.work.device)
        self.delta = None if delta is None else torch.tensor(delta, device=self.work.device, dtype=torch.float32).repeat_interleave(2, dim=0)
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
            hidden = self.model.model(ids, attention_mask=attention, use_cache=False).last_hidden_state
            last = hidden[torch.arange(len(hidden), device=self.work.device), attention.sum(1)-1]
            logits = self.model.lm_head(last).float()
            letter_logits = logits[:, self.answers]
            labels = torch.tensor([row['first_label'] if task == 'first' else row['last_label'] for row in rows for task in ('first', 'last')], device=self.work.device)
            correct = logits[torch.arange(len(logits), device=self.work.device), self.answers[labels]]
            log_probability = correct - torch.logsumexp(logits, dim=1)
            word_hidden = self.captured.reshape(len(rows), 2, -1)
            shared_error = float((word_hidden[:, 0]-word_hidden[:, 1]).abs().max())
            result = dict(logits26=letter_logits.cpu().numpy().reshape(len(rows), 2, 26), logprobs26=(letter_logits-torch.logsumexp(logits, dim=1, keepdim=True)).cpu().numpy().reshape(len(rows), 2, 26), correct_log_probability=log_probability.cpu().numpy().reshape(len(rows), 2), full_vocab_prediction=logits.argmax(1).cpu().numpy().reshape(len(rows), 2), word_hidden=word_hidden[:, 0].cpu().numpy(), prefix_max_absolute_error=np.array(shared_error))
        self.work.sequence_forwards += len(prompts)
        self.work.token_forwards += int(attention.sum())
        if time.perf_counter()-self.work.wall_start > self.config['budget_seconds']:
            raise TimeoutError('Functional fragment driver budget exceeded; completed blocks retained')
        return result

    def block_path(self, name, ids):
        path = self.work.run/'blocks'/name
        if self.previous is not None:
            previous = self.previous/'blocks'/name
            if previous.exists():
                with np.load(previous) as saved:
                    assert str(saved['status']) == 'PASS' and str(saved['identity']) == self.identity_string
                    np.testing.assert_array_equal(saved['word_family_sha256'], ids)
                self.work.checked(previous, 'Completed functional fragment block')
                return previous, True
        return path, False

    def save_block(self, path, rows, **arrays):
        temporary = path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, status='PASS', identity=self.identity_string, word_family_sha256=[row['word_family_sha256'] for row in rows], row_ids=[row['row_id'] for row in rows], first_labels=[row['first_label'] for row in rows], last_labels=[row['last_label'] for row in rows], **arrays)
        temporary.replace(path)


def source_phase(work, config):
    runner = FragmentRunner(work, config)
    (work.run/'blocks').mkdir()
    index = dict(identity=runner.identity, capture_blocks={}, response_blocks=[], operations=list(OPERATIONS), task_axis=['first', 'last'])
    write(work.run/'response_index.json', index)
    maximum_shared_error = 0.
    try:
        for split in ('mean', 'fit', 'calibration', 'development'):
            rows = [row for row in runner.rows if row['split'] == split]
            index['capture_blocks'][split] = []
            for start in range(0, len(rows), config['batch_size']):
                batch = rows[start:start+config['batch_size']]
                path, reused = runner.block_path(f'capture_{split}_{start:05d}.npz', [row['word_family_sha256'] for row in batch])
                if not reused:
                    observed = runner.forward(batch)
                    hidden = observed.pop('word_hidden')
                    with torch.no_grad():
                        codes = runner.saes[1].encode(torch.tensor(hidden, device=work.device)).cpu().numpy()
                        seed_codes = {f'seed{seed}_codes': sae.encode(torch.tensor(hidden, device=work.device)).cpu().numpy() for seed, sae in runner.saes.items()} if split == 'mean' else {}
                    reconstruction = codes @ runner.decoder + runner.source_bias
                    runner.save_block(path, batch, hidden=hidden, source_codes=codes, source_reconstruction=reconstruction, **seed_codes, **observed)
                with np.load(path) as saved:
                    maximum_shared_error = max(maximum_shared_error, float(saved['prefix_max_absolute_error']))
                index['capture_blocks'][split].append(str(path.resolve()))
                write(work.run/'response_index.json', index)
                work.record(kind='fragment_capture', task=split, row_id=start, component=f'{split}_{start}', method='source1', mode='capture', operation='clean', seed=1, path=str(path.resolve()), reused=reused)
                work.progress('CAPTURE', split=split, completed_words=start+len(batch), words=len(rows), reused=reused)
        # 拟合只读取独立 mean 与 labels-fit。
        loaded = {}
        for split in ('mean', 'fit'):
            pieces = []
            for path in index['capture_blocks'][split]:
                with np.load(path) as saved:
                    pieces.append({key: saved[key] for key in saved.files})
            loaded[split] = {key: np.concatenate([piece[key] for piece in pieces]) for key in ('source_reconstruction', 'source_codes', 'first_labels', 'last_labels', 'word_family_sha256')}
            if split == 'mean':
                means = {f'seed{seed}_mean_codes': np.concatenate([piece[f'seed{seed}_codes'] for piece in pieces]).mean(0, dtype=np.float64) for seed in runner.saes}
                means['mean_hidden'] = np.concatenate([piece['hidden'] for piece in pieces]).mean(0, dtype=np.float64)
                means['mean_reconstruction'] = loaded[split]['source_reconstruction'].mean(0, dtype=np.float64)
                np.savez_compressed(work.run/'independent_means.npz', **means, word_family_sha256=loaded[split]['word_family_sha256'])
        mean, fit = loaded['mean'], loaded['fit']
        work.progress('FIT_SOURCE_BASIS', mean_words=len(mean['word_family_sha256']), fit_words=len(fit['word_family_sha256']))
        basis = fit_basis(fit['source_reconstruction'], fit['first_labels'], fit['last_labels'], mean['source_reconstruction'], mean['first_labels'], mean['last_labels'], mean['source_codes'], fit['word_family_sha256'].tolist(), mean['word_family_sha256'].tolist(), input_identity=runner.identity)
        basis_directory = work.run/'basis'
        save_basis(basis, basis_directory)
        index.update(basis_directory=str(basis_directory.resolve()), independent_means=str((work.run/'independent_means.npz').resolve()))
        write(work.run/'response_index.json', index)
        del loaded, fit, mean
        for split in config.get('source_splits', ['calibration', 'development']):
            rows = [row for row in runner.rows if row['split'] == split]
            for block_index, start in enumerate(range(0, len(rows), config['batch_size'])):
                batch = rows[start:start+config['batch_size']]
                with np.load(index['capture_blocks'][split][block_index]) as saved:
                    codes = saved['source_codes']
                    baseline = {key:saved[key] for key in ('logits26', 'logprobs26', 'correct_log_probability', 'full_vocab_prediction')}
                carriers = {role:carrier_masks(codes, runner.decoder, basis, role, k=4) for role in ('F', 'L')}
                for operation in OPERATIONS:
                    path, reused = runner.block_path(f'response_{split}_{start:05d}_{operation}.npz', [row['word_family_sha256'] for row in batch])
                    if not reused:
                        if operation == 'clean':
                            delta = np.zeros((len(batch), runner.decoder.shape[1]), np.float32)
                            observed = baseline
                        else:
                            role = operation.split('_')[0]
                            carrier_ids = carriers[role]['ids'] if 'carrier4' in operation else None
                            delta = operation_delta(codes, runner.decoder, basis, role, carrier_ids=carrier_ids, whole_carriers=operation.endswith('_whole'))
                            observed = runner.forward(batch, delta)
                            observed.pop('word_hidden')
                        runner.save_block(path, batch, operation=operation, delta=delta, F_carrier_ids=carriers['F']['ids'], L_carrier_ids=carriers['L']['ids'], F_carrier_energies=carriers['F']['energies'], L_carrier_energies=carriers['L']['energies'], **observed)
                    index['response_blocks'].append(dict(path=str(path.resolve()), split=split, operation=operation, start=start, words=len(batch), reused=reused))
                    write(work.run/'response_index.json', index)
                    work.record(kind='fragment_response', task=split, row_id=start, component=f'{split}_{start}', method='source1', mode='word_position', operation=operation, seed=1, path=str(path.resolve()), words=len(batch), reused=reused)
                work.progress('SOURCE_RESPONSES', split=split, completed_words=start+len(batch), words=len(rows))
        evaluation_rows = [row for split in config.get('source_splits', ['calibration', 'development']) for row in runner.rows if row['split'] == split]
        row_lookup = {row['word_family_sha256']:i for i, row in enumerate(evaluation_rows)}
        shape = (len(evaluation_rows), len(OPERATIONS), 2)
        letter_logits, letter_logprobs = np.empty((*shape, 26), np.float32), np.empty((*shape, 26), np.float32)
        predictions = np.empty(shape, np.int64)
        carrier_F, carrier_L = np.empty((len(evaluation_rows), 4), np.int64), np.empty((len(evaluation_rows), 4), np.int64)
        for record in index['response_blocks']:
            operation_index = OPERATIONS.index(record['operation'])
            with np.load(record['path']) as saved:
                ix = [row_lookup[str(value)] for value in saved['word_family_sha256']]
                letter_logits[ix, operation_index] = saved['logits26']
                letter_logprobs[ix, operation_index] = saved['logprobs26']
                predictions[ix, operation_index] = saved['full_vocab_prediction']
                carrier_F[ix], carrier_L[ix] = saved['F_carrier_ids'], saved['L_carrier_ids']
        np.savez_compressed(work.run/'source_results.npz', letter_logits=letter_logits, letter_logprobs=letter_logprobs, full_vocab_prediction=predictions, carrier_ids_F=carrier_F, carrier_ids_L=carrier_L, carrier_valid_F=carrier_F>=0, carrier_valid_L=carrier_L>=0, operation_names=OPERATIONS, task_names=['first', 'last'], first_labels=[row['first_label'] for row in evaluation_rows], last_labels=[row['last_label'] for row in evaluation_rows], word_family_sha256=[row['word_family_sha256'] for row in evaluation_rows], split=[row['split'] for row in evaluation_rows], answer_ids=runner.panel['answer_ids'])
        index['source_results'] = str((work.run/'source_results.npz').resolve())
        write(work.run/'response_index.json', index)
        with np.load(work.run/'source_results.npz') as saved:
            calibration = saved['split'] == 'calibration'
            calibration_data = {key:saved[key][calibration] for key in ('letter_logits', 'letter_logprobs', 'first_labels', 'last_labels', 'word_family_sha256')}
            calibration_data.update(operation_names=saved['operation_names'], task_names=saved['task_names'])
        scales = calibration_scales(calibration_data)
        selected_rows = audit_rows([row for row in evaluation_rows if row['split'] == 'development'], config.get('audit_words', {}).get('development', 32))
        selected_ids = {row['word_family_sha256'] for row in selected_rows}
        index.update(audit_blocks=[], calibration_scales=scales.tolist())
        write(work.run/'audit_panel.json', dict(word_family_sha256=[row['word_family_sha256'] for row in selected_rows], selection='Within first-letter strata SHA order, round-robin A-Z; no model outcomes'))
        for path in index['capture_blocks']['development']:
            with np.load(path) as saved:
                captures = {key:saved[key] for key in ('word_family_sha256', 'source_codes', 'logits26')}
            for i, family in enumerate(captures['word_family_sha256']):
                if str(family) in selected_ids:
                    row = next(row for row in selected_rows if row['word_family_sha256'] == str(family))
                    target_audit(work, config, runner, row, captures['source_codes'][i], runner.decoder, basis.mean_codes, captures['logits26'][i], scales, index)
        work.checks['complete_carrier_audit'] = len(index['audit_blocks']) == len(selected_ids)
        work.checks.update(disjoint_word_families=len({row['word_family_sha256'] for row in runner.rows})==len(runner.rows), independent_mean=True, source_only_fit=True, confirmation_not_evaluated=True, complete_source_operations=True, shared_prefix_state=maximum_shared_error < 1e-4)
        write(work.run/'source_checks.json', dict(maximum_shared_prefix_absolute_error=maximum_shared_error, mean_families=sum(row['split']=='mean' for row in runner.rows), basis_directory=str(basis_directory), scope='Single word location; same prefix; SAE residual retained; source phase only'))
    finally:
        runner.handle.remove()


def audit_rows(rows, count):
    groups = {label: sorted([row for row in rows if row['first_label'] == label], key=lambda row:row['word_family_sha256']) for label in range(26)}
    chosen, offset = [], 0
    while len(chosen) < min(count, len(rows)):
        for label in range(26):
            if offset < len(groups[label]) and len(chosen) < count:
                chosen.append(groups[label][offset])
        offset += 1
    return chosen


def target_audit(work, config, runner, row, codes, decoder, mean_codes, baseline, scales, index):
    member_ids = np.flatnonzero(codes > 0)
    name = f'audit_{row["split"]}_{row["row_id"]:05d}.npz'
    path, reused = runner.block_path(name, [row['word_family_sha256']])
    if not reused:
        logits, logprobs, predictions = [], [], []
        for start in range(0, len(member_ids), config.get('audit_member_batch_size', config['batch_size'])):
            selected = member_ids[start:start+config.get('audit_member_batch_size', config['batch_size'])]
            delta = -(codes[selected]-mean_codes[selected])[:, None]*decoder[selected]
            observed = runner.forward([row]*len(selected), delta)
            logits.append(observed['logits26'])
            logprobs.append(observed['logprobs26'])
            predictions.append(observed['full_vocab_prediction'])
        runner.save_block(path, [row], member_ids=member_ids, letter_logits=np.concatenate(logits), letter_logprobs=np.concatenate(logprobs), full_vocab_prediction=np.concatenate(predictions), baseline_logits=baseline, natural_active_count=len(member_ids))
    with np.load(path) as saved:
        member_ids, letter_logits = saved['member_ids'], saved['letter_logits']
    labels = np.array([row['first_label'], row['last_label']])
    margins = (26*np.take_along_axis(letter_logits, labels[None, :, None], axis=2)[..., 0]-letter_logits.sum(2))/25
    base_margin = (26*baseline[np.arange(2), labels]-baseline.sum(1))/25
    delta_margins = margins-base_margin
    joint_paths = []
    for role in ('F', 'L'):
        selected = singleton_oracle(member_ids, delta_margins, scales, role)
        joint_path, joint_reused = runner.block_path(f'audit_joint_{row["split"]}_{row["row_id"]:05d}_{role}.npz', [row['word_family_sha256']])
        if not joint_reused:
            chosen = np.asarray(selected['ids'], dtype=np.int64)
            if selected['defined']:
                delta = -((codes[chosen]-mean_codes[chosen]) @ decoder[chosen])[None, :]
                observed = runner.forward([row], delta)
                observed.pop('word_hidden')
                runner.save_block(joint_path, [row], member_ids=chosen, role=role, oracle_defined=True, singleton_utilities=selected['utilities'], delta=delta, **observed)
            else:
                runner.save_block(joint_path, [row], member_ids=chosen, role=role, oracle_defined=False, singleton_utilities=selected['utilities'])
        joint_paths.append(str(joint_path.resolve()))
    index['audit_blocks'].append(dict(path=str(path.resolve()), oracle_joint_paths=joint_paths, split=row['split'], word_family_sha256=row['word_family_sha256'], active_members=len(member_ids), reused=reused))
    write(work.run/'response_index.json', index)
    dictionary_seed = config.get('target_seed', 1)
    work.record(kind='carrier_audit', task=row['split'], row_id=row['row_id'], component=row['word_family_sha256'], method=f'seed{dictionary_seed}_native_singleton', mode='all_natural_active', operation='singleton_and_oracle_joint', seed=dictionary_seed, target_seed=dictionary_seed, members=len(member_ids), path=str(path.resolve()), reused=reused)
    work.progress('CARRIER_AUDIT', split=row['split'], completed_words=len(index['audit_blocks']), active_members=len(member_ids))


def target_phase(work, config):
    runner = FragmentRunner(work, config)
    (work.run/'blocks').mkdir()
    source_run = Path(config['source_run'])
    source_identity = json.loads(work.checked(source_run/'identity.json').read_text())
    for key in ('panel_sha256', 'model_revision', 'hook'):
        assert source_identity[key] == runner.identity[key]
    for seed in runner.saes:
        assert source_identity['checkpoint_identities'][str(seed)] == runner.identity['checkpoint_identities'][str(seed)]
    basis_path = work.checked(source_run/'basis/basis.npz')
    work.checked(source_run/'basis/BASIS.json')
    basis = load_basis(basis_path.parent)
    with np.load(work.checked(source_run/'independent_means.npz')) as saved:
        mean_codes = saved[f'seed{config["target_seed"]}_mean_codes']
        mean_hidden = saved['mean_hidden']
    with np.load(work.checked(source_run/'source_results.npz')) as saved:
        calibration = saved['split'] == 'calibration'
        calibration_data = {key: saved[key][calibration] for key in ('letter_logits', 'letter_logprobs', 'first_labels', 'last_labels', 'word_family_sha256')}
        calibration_data['operation_names'] = saved['operation_names']
        calibration_data['task_names'] = saved['task_names']
    scales = calibration_scales(calibration_data)
    operations = (*OPERATIONS, 'whole_energy4', 'PW_F_carrier4_whole', 'PW_L_carrier4_whole', 'raw_F', 'raw_L', 'raw_F+L')
    decoder = runner.saes[config['target_seed']].decoder.weight.T.detach().cpu().numpy()
    index = dict(identity=runner.identity, source_run=str(source_run), source_basis_sha256=sha256(basis_path), capture_blocks=[], response_blocks=[], audit_blocks=[], operations=operations, task_axis=['first', 'last'], calibration_scales=scales.tolist())
    reference_operations = ('clean', 'F_carrier4_whole', 'L_carrier4_whole')
    collect_reference = config.get('collect_source_reference', False)
    assert not (collect_reference and config.get('source_reference_run'))
    if collect_reference:
        index['source_reference_blocks'] = []
    elif config.get('source_reference_run'):
        reference_run = Path(config['source_reference_run'])
        reference_index = json.loads(work.checked(reference_run/'response_index.json').read_text())
        for key in ('panel_sha256', 'model_revision', 'hook', 'batch_size', 'matmul_precision'):
            assert reference_index['identity'][key] == runner.identity[key]
        assert reference_index['identity']['checkpoint_identities']['1'] == runner.identity['checkpoint_identities']['1']
        assert reference_index['source_basis_sha256'] == sha256(basis_path)
        assert sha256(Path(reference_index['source_run'])/'independent_means.npz') == sha256(source_run/'independent_means.npz')
        reference_path = work.checked(reference_index['source_reference_results'])
        with np.load(reference_path) as saved:
            for key, expected in [('word_family_sha256', [row['word_family_sha256'] for row in runner.rows]), ('first_labels', [row['first_label'] for row in runner.rows]), ('last_labels', [row['last_label'] for row in runner.rows]), ('split', [row['split'] for row in runner.rows]), ('answer_ids', runner.panel['answer_ids']), ('operation_names', reference_operations)]:
                np.testing.assert_array_equal(saved[key], expected)
        index.update(source_reference_results=str(reference_path.resolve()), source_reference_reused_from=str(reference_run.resolve()), source_reference_blocks=reference_index['source_reference_blocks'])
    write(work.run/'response_index.json', index)
    audit_ids = set()
    for split, count in [('development', 32), ('confirmation', 64)]:
        selected = audit_rows([row for row in runner.rows if row['split'] == split], config.get('audit_words', {}).get(split, count))
        audit_ids.update(row['word_family_sha256'] for row in selected)
    write(work.run/'audit_panel.json', dict(word_family_sha256=sorted(audit_ids), selection='Within first-letter strata SHA order, round-robin A-Z; no model outcomes'))
    maximum_shared_error = 0.
    try:
        for start in range(0, len(runner.rows), config['batch_size']):
            rows = runner.rows[start:start+config['batch_size']]
            capture_path, reused = runner.block_path(f'target_capture_{start:05d}.npz', [row['word_family_sha256'] for row in rows])
            if not reused:
                observed = runner.forward(rows)
                hidden = observed.pop('word_hidden')
                with torch.no_grad():
                    tensor = torch.tensor(hidden, device=work.device)
                    source_codes = runner.saes[1].encode(tensor).cpu().numpy()
                    codes = runner.saes[config['target_seed']].encode(tensor).cpu().numpy()
                runner.save_block(capture_path, rows, hidden=hidden, source_codes=source_codes, target_codes=codes, **observed)
            with np.load(capture_path) as saved:
                codes, source_codes, hidden = saved['target_codes'], saved['source_codes'], saved['hidden']
                baseline = {key:saved[key] for key in ('logits26', 'logprobs26', 'correct_log_probability', 'full_vocab_prediction')}
                maximum_shared_error = max(maximum_shared_error, float(saved['prefix_max_absolute_error']))
            index['capture_blocks'].append(str(capture_path.resolve()))
            carriers = {role:carrier_masks(codes, decoder, basis, role, mean_codes=mean_codes, k=4) for role in ('F', 'L')}
            source_carriers = {role:carrier_masks(source_codes, runner.decoder, basis, role, k=4) for role in ('F', 'L')}
            matched = {role:match_source_carriers(source_carriers[role]['ids'], runner.decoder, decoder, target_codes=codes) for role in ('F', 'L')}
            energy_carriers = whole_energy_carriers(codes, decoder, basis, mean_codes=mean_codes)
            if collect_reference:
                for operation in reference_operations:
                    path, reference_reused = runner.block_path(f'source_reference_{start:05d}_{operation}.npz', [row['word_family_sha256'] for row in rows])
                    if not reference_reused:
                        if operation == 'clean':
                            reference_delta, reference_observed = np.zeros_like(hidden), baseline
                        else:
                            role = operation[0]
                            reference_delta = operation_delta(source_codes, runner.decoder, basis, role, carrier_ids=source_carriers[role]['ids'], whole_carriers=True)
                            reference_observed = runner.forward(rows, reference_delta)
                            reference_observed.pop('word_hidden')
                        runner.save_block(path, rows, operation=operation, delta=reference_delta, F_carrier_ids=source_carriers['F']['ids'], L_carrier_ids=source_carriers['L']['ids'], **reference_observed)
                    index['source_reference_blocks'].append(dict(path=str(path.resolve()), operation=operation, start=start, words=len(rows), reused=reference_reused, additional_sequence_forwards=0 if reference_reused or operation=='clean' else 2*len(rows)))
                    write(work.run/'response_index.json', index)
                    work.record(kind='source_reference_response', task='source_reference', row_id=start, component=f'block_{start}', method='source1', mode='word_position', operation=operation, seed=1, path=str(path.resolve()), words=len(rows), reused=reference_reused)
            for operation in operations:
                path, reused = runner.block_path(f'target_response_{start:05d}_{operation}.npz', [row['word_family_sha256'] for row in rows])
                if not reused:
                    if operation == 'clean':
                        delta, observed = np.zeros_like(hidden), baseline
                    else:
                        if operation.startswith('raw_'):
                            delta = -role_projection(hidden-mean_hidden, basis, operation[4:])
                        elif operation.startswith('PW_'):
                            role = operation.split('_')[1]
                            delta = operation_delta(codes, decoder, basis, role, mean_codes=mean_codes, carrier_ids=matched[role]['ids'], whole_carriers=True)
                        elif operation == 'whole_energy4':
                            delta = operation_delta(codes, decoder, basis, 'F', mean_codes=mean_codes, carrier_ids=energy_carriers['ids'], whole_carriers=True)
                        else:
                            role = operation.split('_')[0]
                            selected = carriers[role]['ids'] if 'carrier4' in operation else None
                            delta = operation_delta(codes, decoder, basis, role, mean_codes=mean_codes, carrier_ids=selected, whole_carriers=operation.endswith('_whole'))
                        observed = runner.forward(rows, delta)
                        observed.pop('word_hidden')
                    runner.save_block(path, rows, operation=operation, delta=delta, F_carrier_ids=carriers['F']['ids'], L_carrier_ids=carriers['L']['ids'], whole_energy_ids=energy_carriers['ids'], PW_F_ids=matched['F']['ids'], PW_L_ids=matched['L']['ids'], PW_F_inactive_count=matched['F']['inactive_count'], PW_L_inactive_count=matched['L']['inactive_count'], source_F_carrier_ids=source_carriers['F']['ids'], source_L_carrier_ids=source_carriers['L']['ids'], **observed)
                index['response_blocks'].append(dict(path=str(path.resolve()), operation=operation, start=start, words=len(rows), reused=reused))
                write(work.run/'response_index.json', index)
                work.record(kind='fragment_response', task='target', row_id=start, component=f'block_{start}', method=f'target{config["target_seed"]}', mode='word_position', operation=operation, seed=1, target_seed=config['target_seed'], path=str(path.resolve()), words=len(rows), reused=reused)
            for i, row in enumerate(rows):
                if row['word_family_sha256'] in audit_ids:
                    target_audit(work, config, runner, row, codes[i], decoder, mean_codes, baseline['logits26'][i], scales, index)
            work.progress('TARGET_RESPONSES', target_seed=config['target_seed'], completed_words=start+len(rows), words=len(runner.rows))
        row_lookup = {row['word_family_sha256']:i for i, row in enumerate(runner.rows)}
        shape = (len(runner.rows), len(operations), 2)
        logits, logprobs, predictions = np.empty((*shape, 26), np.float32), np.empty((*shape, 26), np.float32), np.empty(shape, np.int64)
        for record in index['response_blocks']:
            operation_index = operations.index(record['operation'])
            with np.load(record['path']) as saved:
                ix = [row_lookup[str(value)] for value in saved['word_family_sha256']]
                logits[ix, operation_index], logprobs[ix, operation_index], predictions[ix, operation_index] = saved['logits26'], saved['logprobs26'], saved['full_vocab_prediction']
        np.savez_compressed(work.run/'target_results.npz', letter_logits=logits, letter_logprobs=logprobs, full_vocab_prediction=predictions, operation_names=operations, task_names=['first', 'last'], first_labels=[row['first_label'] for row in runner.rows], last_labels=[row['last_label'] for row in runner.rows], word_family_sha256=[row['word_family_sha256'] for row in runner.rows], split=[row['split'] for row in runner.rows], answer_ids=runner.panel['answer_ids'], target_seed=config['target_seed'], calibration_scales=scales)
        index['target_results'] = str((work.run/'target_results.npz').resolve())
        if collect_reference:
            shape = (len(runner.rows), len(reference_operations), 2)
            reference_logits, reference_logprobs = np.empty((*shape, 26), np.float32), np.empty((*shape, 26), np.float32)
            reference_predictions = np.empty(shape, np.int64)
            reference_F, reference_L = np.empty((len(runner.rows), 4), np.int64), np.empty((len(runner.rows), 4), np.int64)
            for record in index['source_reference_blocks']:
                operation_index = reference_operations.index(record['operation'])
                with np.load(record['path']) as saved:
                    ix = [row_lookup[str(value)] for value in saved['word_family_sha256']]
                    reference_logits[ix, operation_index], reference_logprobs[ix, operation_index], reference_predictions[ix, operation_index] = saved['logits26'], saved['logprobs26'], saved['full_vocab_prediction']
                    reference_F[ix], reference_L[ix] = saved['F_carrier_ids'], saved['L_carrier_ids']
            reference_path = work.run/'source_reference_results.npz'
            np.savez_compressed(reference_path, letter_logits=reference_logits, letter_logprobs=reference_logprobs, full_vocab_prediction=reference_predictions, carrier_ids_F=reference_F, carrier_ids_L=reference_L, operation_names=reference_operations, task_names=['first', 'last'], first_labels=[row['first_label'] for row in runner.rows], last_labels=[row['last_label'] for row in runner.rows], word_family_sha256=[row['word_family_sha256'] for row in runner.rows], split=[row['split'] for row in runner.rows], answer_ids=runner.panel['answer_ids'], calibration_scales=scales)
            index['source_reference_results'] = str(reference_path.resolve())
            work.checks['complete_source_reference'] = len(index['source_reference_blocks']) == 3*len(index['capture_blocks'])
        write(work.run/'response_index.json', index)
        work.checks.update(frozen_source_basis=True, independent_target_mean=True, no_target_response_fitting=True, complete_target_operations=True, complete_carrier_audit=len(index['audit_blocks'])==len(audit_ids), shared_prefix_state=maximum_shared_error<1e-4)
    finally:
        runner.handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config['phase'] in ('source', 'target')
    work = MultisiteWork(config, args.config, ['scripts/run_functional_fragments.py', 'scripts/prepare_fragment_words.py', 'scripts/functional_fragment_model.py', 'scripts/run_causalgym_multisite.py', 'scripts/run_r011s1_raw_hook_asset.py', 'src/ccad/artifacts.py'])
    error = None
    try:
        if config['phase'] == 'source':
            source_phase(work, config)
        else:
            target_phase(work, config)
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
