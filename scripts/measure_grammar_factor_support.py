import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

from run_causalgym_multisite import MultisiteWork, write
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor, replace_primary_hook_tensor
from ccad.artifacts import sha256

import numpy as np
import torch
import transformers


WORDS = [' is', ' are', ' was', ' were']


def load_config(path):
    current = json.loads(Path(path).read_text(encoding='utf-8'))
    result = load_config(current['base_config']) if current.get('base_config') else {}
    result.update(current)
    return result


def summarize(logp, token_ids, numbers, tenses, baseline=None, axis=None):
    selected = logp[:, token_ids]
    stable = np.exp(selected - selected.max(axis=1, keepdims=True))
    conditional = stable / stable.sum(axis=1, keepdims=True)
    number_probability = np.stack([conditional[:, [0, 2]].sum(1), conditional[:, [1, 3]].sum(1)], axis=1)
    tense_probability = np.stack([conditional[:, :2].sum(1), conditional[:, 2:].sum(1)], axis=1)
    number_prediction, tense_prediction = number_probability.argmax(1), tense_probability.argmax(1)
    expected_number = 1-numbers if axis == 'number' else numbers
    expected_tense = 1-tenses if axis == 'tense' else tenses
    number_correct, tense_correct = number_prediction == expected_number, tense_prediction == expected_tense
    arrays = dict(selected_vocab_logprobs=selected, four_word_conditional_probability=conditional,
                  four_word_vocab_probability=np.exp(selected), four_word_vocab_mass=np.exp(selected).sum(1),
                  four_word_argmax=conditional.argmax(1), vocab_argmax=logp.argmax(1),
                  number_probability=number_probability, tense_probability=tense_probability,
                  number_prediction=number_prediction, tense_prediction=tense_prediction,
                  expected_number=expected_number, expected_tense=expected_tense,
                  number_correct=number_correct, tense_correct=tense_correct,
                  joint_success=number_correct & tense_correct,
                  four_word_correct=conditional.argmax(1) == expected_number+2*expected_tense)
    summary = dict(documents=len(logp), number_accuracy=float(number_correct.mean()),
                   tense_accuracy=float(tense_correct.mean()), joint_success=float(arrays['joint_success'].mean()),
                   four_word_accuracy=float(arrays['four_word_correct'].mean()),
                   mean_four_word_conditional_probability=conditional.mean(0).tolist(),
                   mean_four_word_vocab_probability=np.exp(selected).mean(0).tolist(),
                   mean_four_word_vocab_mass=float(arrays['four_word_vocab_mass'].mean()))
    if axis is not None:
        other = 'tense' if axis == 'number' else 'number'
        summary['target_axis_hit_rate'] = float(arrays[axis+'_correct'].mean())
        summary['non_target_axis_semantic_retention'] = float(arrays[other+'_correct'].mean())
        retained = arrays[other+'_prediction'] == baseline[other+'_prediction']
        arrays['non_target_prediction_retained'] = retained
        summary['non_target_baseline_prediction_retention'] = float(retained.mean())
        summary['target_hit_and_baseline_other_retained'] = float((arrays[axis+'_correct'] & retained).mean())
    return arrays, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    work = MultisiteWork(config, args.config, [
        'scripts/measure_grammar_factor_support.py', 'scripts/run_causalgym_multisite.py',
        'scripts/run_causalgym_native_transfer.py', 'scripts/run_r011s1_raw_hook_asset.py',
        'src/ccad/artifacts.py', 'src/ccad/activation_contract.py'])
    error = None
    padded_token_forwards = 0
    try:
        work.torch, work.device = torch, torch.device(config['device'])
        torch.set_num_threads(config.get('cpu_threads', 2))
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        if work.device.type == 'cuda':
            torch.cuda.set_device(work.device)
            torch.cuda.reset_peak_memory_stats(work.device)
        panel_path = work.checked(config['panel_path'], 'Predeclared factorial grammar inputs')
        panel = json.loads(panel_path.read_text(encoding='utf-8'))
        rows = panel['rows']
        assert rows
        assert config.get('labels', WORDS) == WORDS
        keys = [(str(row['block']), int(row['number']), int(row['tense']), int(row['attractor'])) for row in rows]
        assert len(set(keys)) == len(keys)
        assert all(n in (0, 1) and t in (0, 1) and a in (0, 1) for _, n, t, a in keys)
        lookup = {key: index for index, key in enumerate(keys)}
        donors = dict(number=np.array([lookup[(b, 1-n, t, a)] for b, n, t, a in keys]),
                      tense=np.array([lookup[(b, n, 1-t, a)] for b, n, t, a in keys]))
        for indices in donors.values():
            np.testing.assert_array_equal(indices[indices], np.arange(len(rows)))
        model_directory = Path(config['model_local_dir'])
        for filename in ('config.json', 'tokenizer.json', 'model.safetensors'):
            work.checked(model_directory/filename, 'Pinned GPT2-medium '+config['model_revision'])
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_directory, local_files_only=True, trust_remote_code=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        model = transformers.AutoModelForCausalLM.from_pretrained(model_directory, local_files_only=True,
            trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False)
        model.config.use_cache = False
        assert model.config.hidden_size == 1024
        hook_name = config.get('hook_module_path', 'transformer.h.11')
        assert hook_name == 'transformer.h.11'
        module = model.get_submodule(hook_name)
        contract = HookPointContract(hook_name, 11, 'resid_post', 1024)
        tokenized_words = [tokenizer.encode(word, add_special_tokens=False) for word in WORDS]
        assert all(len(value) == 1 for value in tokenized_words)
        token_ids = np.array([value[0] for value in tokenized_words])
        for row in rows:
            row['tokens'] = tokenizer.encode(row['text'], add_special_tokens=False)
            assert 0 < len(row['tokens']) <= config.get('max_length', model.config.max_position_embeddings)
            for word, ids in zip(WORDS, tokenized_words):
                assert tokenizer.encode(row['text']+word, add_special_tokens=False) == row['tokens']+ids
        write(work.run/'panel.json', dict(panel, rows=rows, continuation_tokens=WORDS,
              continuation_ids=token_ids.tolist(), number_values=['singular', 'plural'], tense_values=['present', 'past']))
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
            device=str(work.device), cpu_threads=torch.get_num_threads(), hook=hook_name,
            matmul_precision=torch.get_float32_matmul_precision(),
            gpu=torch.cuda.get_device_name(work.device) if work.device.type == 'cuda' else 'not_used')
        write(work.run/'environment.json', work.environment)
        batch_size = int(config['batch_size'])
        def forward(indices, delta=None):
            nonlocal padded_token_forwards
            if time.perf_counter()-work.wall_start > config['budget_seconds']:
                raise TimeoutError('Declared grammar support budget exhausted')
            encoded = tokenizer([rows[index]['text'] for index in indices], add_special_tokens=False,
                                padding=True, return_tensors='pt').to(work.device)
            batch_ids = torch.arange(len(indices), device=work.device)
            last = encoded.attention_mask.sum(1)-1
            captured = []
            def hook(module, arguments, output):
                hidden = extract_primary_hook_tensor(output, contract)
                captured.append(hidden[batch_ids, last].detach().cpu().numpy())
                if delta is None:
                    return output
                changed = hidden.clone()
                changed[batch_ids, last] += torch.as_tensor(delta, device=work.device, dtype=hidden.dtype)
                return replace_primary_hook_tensor(output, changed, contract)
            handle = module.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    logits = model(**encoded, use_cache=False).logits[batch_ids, last]
                    logp = logits.double().log_softmax(-1).cpu().numpy()
            finally:
                handle.remove()
            assert len(captured) == 1
            work.sequence_forwards += len(indices)
            work.token_forwards += int(encoded.attention_mask.sum())
            padded_token_forwards += int(encoded.input_ids.numel())
            return logp, captured[0]
        indices = np.arange(len(rows))
        baseline = np.empty((len(rows), model.config.vocab_size), dtype=np.float64)
        hidden = np.empty((len(rows), 1024), dtype=np.float32)
        for start in range(0, len(rows), batch_size):
            chosen = indices[start:start+batch_size]
            baseline[chosen], hidden[chosen] = forward(chosen)
            work.progress('BASELINE_CAPTURE', documents_completed=start+len(chosen), documents_total=len(rows))
        probe = indices[:batch_size]
        noop, recaptured = forward(probe, np.zeros_like(hidden[probe]))
        np.testing.assert_array_equal(noop, baseline[probe])
        np.testing.assert_array_equal(recaptured, hidden[probe])
        work.checks.update(noop_replay_exact=True, captured_hidden_exact=True, reciprocal_donors=True,
                           declared_inputs_only=True, fixed_last_token=True)
        numbers, tenses = np.array([row['number'] for row in rows]), np.array([row['tense'] for row in rows])
        common = dict(token_ids=token_ids, continuation_tokens=np.array(WORDS), numbers=numbers, tenses=tenses,
                      attractors=np.array([row['attractor'] for row in rows]), blocks=np.array([str(row['block']) for row in rows]),
                      number_donors=donors['number'], tense_donors=donors['tense'])
        baseline_arrays, baseline_summary = summarize(baseline, token_ids, numbers, tenses)
        np.savez_compressed(work.run/'baseline.npz', **common, **baseline_arrays, hidden=hidden)
        summaries = dict(baseline=baseline_summary)
        method_states = dict(raw=hidden)
        decoder = None
        if config.get('source_checkpoint'):
            sys.path.extend([config['dictionary_source_dir'], config['dictionary_overlay_dir']])
            from dictionary_learning.trainers.top_k import AutoEncoderTopK
            work.checked(Path(config['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py', 'Pinned TopK SAE implementation', 'MIT')
            checkpoint = work.checked(config['source_checkpoint'])
            if config.get('source_checkpoint_sha256'):
                assert sha256(checkpoint) == config['source_checkpoint_sha256']
            state = torch.load(checkpoint, map_location=work.device, weights_only=True)
            assert state['encoder.weight'].shape == (8192, 1024) and int(state['k']) == 64
            sae = AutoEncoderTopK(1024, 8192, 64).to(work.device)
            sae.load_state_dict(state)
            sae.eval().requires_grad_(False)
            with torch.no_grad():
                method_states['source_sae'] = sae.encode(torch.as_tensor(hidden, device=work.device)).cpu().numpy()
            decoder = sae.decoder.weight.T.detach()
            np.savez_compressed(work.run/'source_codes.npz', codes=method_states['source_sae'], decoder=decoder.cpu().numpy())
        for method, states in method_states.items():
            for axis in ('number', 'tense'):
                outputs = np.empty_like(baseline)
                delta_values = states[donors[axis]]-states
                if method == 'source_sae':
                    with torch.no_grad():
                        delta_values = (torch.as_tensor(delta_values, device=work.device) @ decoder).cpu().numpy()
                for start in range(0, len(rows), batch_size):
                    chosen = indices[start:start+batch_size]
                    outputs[chosen], _ = forward(chosen, delta_values[chosen])
                    work.progress('AXIS_SWAP', method=method, axis=axis,
                                  documents_completed=start+len(chosen), documents_total=len(rows))
                arrays, summary = summarize(outputs, token_ids, numbers, tenses, baseline_arrays, axis)
                name = method+'_'+axis
                summaries[name] = summary
                np.savez_compressed(work.run/(name+'.npz'), **common, **arrays, delta_hidden=delta_values)
                for index, row in enumerate(rows):
                    work.record(kind='grammar_factor_support', task='number_tense', component=str(row['block']),
                                row_id=index, mode=axis, operation='last_token_donor_delta', method=method,
                                seed=config['seeds'][0], number=row['number'], tense=row['tense'], attractor=row['attractor'],
                                target_axis_hit=bool(arrays[axis+'_correct'][index]),
                                joint_success=bool(arrays['joint_success'][index]),
                                four_word_correct=bool(arrays['four_word_correct'][index]),
                                four_word_probability=arrays['four_word_conditional_probability'][index].tolist())
                write(work.run/'SUPPORT_RESULTS.json', dict(words=WORDS, rows=len(rows), summaries=summaries,
                      intervention='Full donor-minus-recipient code or hidden-state difference added at the last real token',
                      baseline_semantics='number0 singular; number1 plural; tense0 present; tense1 past',
                      non_target_retention='Semantic correctness and preservation of baseline marginal prediction are separately reported'))
        write(work.run/'execution_cost.json', dict(sequence_forwards=work.sequence_forwards,
              real_token_forwards=work.token_forwards, padded_token_forwards=padded_token_forwards,
              noop_sequence_forwards=len(probe), trained_parameters=0))
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
