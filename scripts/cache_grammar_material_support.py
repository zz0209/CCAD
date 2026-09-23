import argparse
from datetime import datetime, timezone
import json
import platform
import sys
import time
import traceback
from pathlib import Path

from run_causalgym_multisite import MultisiteWork, ROOT, write
from ccad.activation_contract import HookPointContract, extract_primary_hook_tensor
from ccad.artifacts import sha256

import numpy as np
import torch
import transformers


def load_config(path):
    current = json.loads(Path(path).read_text(encoding='utf-8'))
    result = load_config(current['base_config']) if current.get('base_config') else {}
    result.update(current)
    return result


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.pending')
    write(temporary, value)
    temporary.replace(path)


def restore_work(config, config_path):
    run = Path(config['run_storage_root'])/config['run_id']
    assert json.loads((run/'config.resolved.json').read_text()) == config
    status = json.loads((run/'status.json').read_text())
    assert status['status'] != 'PASS', 'Completed runs are immutable'
    snapshot = run/'source_snapshot/scripts/cache_grammar_material_support.py'
    assert sha256(snapshot) == sha256(Path(__file__))
    state = json.loads((run/'metrics.summary.json').read_text()) if (run/'metrics.summary.json').exists() else json.loads((run/'progress.json').read_text())
    previous = dict(status=status, state=state, resumed_at_utc=datetime.now(timezone.utc).isoformat())
    if (run/'stderr.log').exists():
        previous['stderr'] = (run/'stderr.log').read_text(encoding='utf-8')
    attempt = len(list(run.glob('resume_attempt_*.json')))+1
    write(run/f'resume_attempt_{attempt:03d}.json', previous)
    work = MultisiteWork.__new__(MultisiteWork)
    work.cfg, work.config_path, work.run = config, Path(config_path), run
    work.started = datetime.fromisoformat(status['started_at_utc'])
    work.wall_start = time.perf_counter()-float(state.get('wall_seconds', state.get('elapsed', 0)))
    work.cpu_start = time.process_time()-float(state.get('process_cpu_seconds', 0))
    work.inputs = json.loads((run/'inputs.json').read_text())['inputs']
    work.metrics = [json.loads(line) for line in (run/'metrics.raw.jsonl').read_text().splitlines()]
    work.checks = state.get('checks', {})
    work.environment = json.loads((run/'environment.json').read_text()) if (run/'environment.json').exists() else {}
    work.sequence_forwards = int(state['sequence_forwards'])
    work.token_forwards = int(state['token_forwards'])
    work.driver = config['generator_script']
    atomic_json(run/'status.json', dict(status='RUNNING', started_at_utc=work.started.isoformat(),
                                       resumed_at_utc=previous['resumed_at_utc']))
    return work


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    config.setdefault('generator_script', 'scripts/cache_grammar_material_support.py')
    sources = ['scripts/cache_grammar_material_support.py', 'scripts/run_causalgym_multisite.py',
               'scripts/run_causalgym_native_transfer.py', 'scripts/run_r011s1_raw_hook_asset.py',
               'src/ccad/artifacts.py', 'src/ccad/activation_contract.py']
    work = restore_work(config, args.config) if args.resume else MultisiteWork(config, args.config, sources)
    error = None
    try:
        work.torch, work.device = torch, torch.device(config['device'])
        torch.set_num_threads(config.get('cpu_threads', 2))
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        model_path = Path(config['model_local_dir'])
        for filename in ('config.json', 'tokenizer.json', 'model.safetensors'):
            work.checked(model_path/filename, 'Pinned Pythia1B model '+config['model_revision'])
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            trust_remote_code=False, dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
        model.requires_grad_(False)
        model.config.use_cache = False
        assert model.config.hidden_size == 2048 and model.config.num_hidden_layers == 16
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        module = model.gpt_neox.layers[15]
        contract = HookPointContract('gpt_neox.layers.15', 15, 'resid_post', 2048)
        work.environment = dict(python=sys.executable, python_version=platform.python_version(), torch=torch.__version__,
            transformers=transformers.__version__, numpy=np.__version__, device=str(work.device),
            gpu=torch.cuda.get_device_name(work.device), cpu_threads=torch.get_num_threads(),
            dtype='float32', matmul_precision='highest', hook='gpt_neox.layers.15', output='pre-final-LayerNorm residual')
        write(work.run/'environment.json', work.environment)
        manifest_path = work.run/'cache_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
            model_revision=config['model_revision'], hook='gpt_neox.layers.15', hidden_size=2048,
            token_rule='Natural and quality inputs use all original 128 tokens in LM attention, including EOS token ID zero; only the saved SAE states omit positions with token ID zero',
            grammar_rule='One final real token state per declared grammar text; no result-based selection')
        input_panel = json.loads(work.checked(config['grammar_panel'], 'Frozen factorial grammar panel').read_text())
        rows = input_panel['rows']
        assert rows
        for index, row in enumerate(rows):
            row['tokens'] = tokenizer.encode(row['text'], add_special_tokens=False)
            row['cache_row_id'] = index
            assert 0 < len(row['tokens']) <= model.config.max_position_embeddings
            for name in ('number', 'time', 'attractor', 'subject_id', 'attractor_id', 'cue_id', 'template'):
                assert name in row
        tokenized_panel_path = work.run/'grammar_panel.json'
        if tokenized_panel_path.exists():
            assert json.loads(tokenized_panel_path.read_text())['rows'] == rows
        else:
            write(tokenized_panel_path, dict(input_panel, rows=rows))
        manifest['grammar_panel'] = dict(path=str(tokenized_panel_path.resolve()), sha256=sha256(tokenized_panel_path), rows=len(rows))
        oracle_done = (work.run/'final_layer_oracle.json').exists()
        if oracle_done:
            assert json.loads((work.run/'final_layer_oracle.json').read_text())['status'] == 'PASS'
        def capture(input_ids, attention_mask):
            nonlocal oracle_done
            if time.perf_counter()-work.wall_start > config['budget_seconds']:
                raise TimeoutError('Declared activation capture budget exhausted')
            observed = []
            def hook(module, arguments, output):
                observed.append(extract_primary_hook_tensor(output, contract).detach())
            handle = module.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    output = model.gpt_neox(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    assert len(observed) == 1
                    hidden = observed[0]
                    assert torch.isfinite(hidden).all()
                    if not oracle_done:
                        error = float((model.gpt_neox.final_layer_norm(hidden)-output.last_hidden_state).abs().max())
                        assert error <= 1e-6, error
                        write(work.run/'final_layer_oracle.json', dict(status='PASS', maximum_absolute_error=error,
                            rows=len(input_ids), positions=int(attention_mask.sum()), tolerance=1e-6,
                            identity='Final LayerNorm applied to the captured block output equals this forward last_hidden_state'))
                        oracle_done = True
                    result = hidden.cpu().numpy()
            finally:
                handle.remove()
            work.sequence_forwards += len(input_ids)
            work.token_forwards += int(attention_mask.sum())
            return result

        def materialize(name, total_inputs, total_outputs, batch_size, input_batch):
            destination = work.run/(name+'.npy')
            shape = (total_outputs, 2048)
            if name in manifest:
                spec = manifest[name]
                existing = np.load(spec['path'], mmap_mode='r', allow_pickle=False)
                assert existing.shape == shape and existing.dtype == np.float32
                assert Path(spec['path']).resolve() == destination.resolve()
                assert spec['shape'] == list(shape) and spec['dtype'] == 'float32'
                work.progress('REUSE_COMPLETED_CACHE', cache=name, states=total_outputs)
                return
            cursor_path = work.run/(name+'_cursor.json')
            if cursor_path.exists():
                cursor = json.loads(cursor_path.read_text())
                assert cursor['input_rows_total'] == total_inputs and cursor['shape'] == list(shape)
                first_input, first_output = cursor['input_rows_completed'], cursor['output_rows_completed']
                cache = np.load(destination, mmap_mode='r+', allow_pickle=False)
                assert cache.shape == shape and cache.dtype == np.float32
            else:
                assert not destination.exists(), str(destination)
                cache = np.lib.format.open_memmap(destination, mode='w+', dtype=np.float32, shape=shape)
                first_input = first_output = 0
                atomic_json(cursor_path, dict(input_rows_completed=0, output_rows_completed=0,
                    input_rows_total=total_inputs, shape=list(shape), dtype='float32'))
            for start in range(first_input, total_inputs, batch_size):
                stop = min(start+batch_size, total_inputs)
                input_ids, mask, positions, selection_mask = input_batch(start, stop)
                values = capture(input_ids, mask)
                selected = values[np.arange(len(values)), positions] if positions is not None else values[selection_mask.cpu().numpy().astype(bool)]
                assert len(selected) > 0
                cache[first_output:first_output+len(selected)] = selected
                first_output += len(selected)
                cache.flush()
                atomic_json(cursor_path, dict(input_rows_completed=stop, output_rows_completed=first_output,
                    input_rows_total=total_inputs, shape=list(shape), dtype='float32'))
                work.progress('CAPTURE_MATERIAL', cache=name, sequences_completed=stop,
                              sequences_total=total_inputs, states_completed=first_output, states_total=total_outputs)
            assert first_output == total_outputs
            cache.flush()
            del cache
            manifest[name] = dict(path=str(destination.resolve()), sha256=sha256(destination), shape=list(shape),
                                  dtype='float32', bytes=destination.stat().st_size, input_sequences=total_inputs)
            atomic_json(manifest_path, manifest)
            work.record(kind='material_cache', task='grammar_material', component=name, row_id=0,
                        method=name, mode='capture', operation='read_hidden', seed=0, states=total_outputs,
                        input_sequences=total_inputs, bytes=destination.stat().st_size)
            work.progress('CACHE_STAGE_COMPLETE', cache=name, states=total_outputs)

        for name, input_key, count_key in [('natural_states', 'natural_token_file', 'natural_sequences'),
                                          ('quality_states', 'quality_token_file', 'quality_sequences')]:
            path = work.checked(config[input_key], 'Original uint16 token sequences')
            tokens = np.memmap(path, mode='r', dtype=np.uint16)
            assert len(tokens) % 128 == 0
            sequences = tokens.reshape(-1, 128)
            count = int(config[count_key])
            assert 0 < count <= len(sequences)
            chosen = sequences[:count]
            assert np.all(np.any(chosen != 0, axis=1))
            total = int(np.count_nonzero(chosen))
            manifest[name+'_input'] = dict(path=str(path.resolve()), sequences=count, tokens_per_sequence=128,
                                           dtype='uint16', selection='first sequences in original order')
            def natural_batch(start, stop):
                ids = torch.as_tensor(np.array(chosen[start:stop], dtype=np.int64), device=work.device)
                return ids, torch.ones_like(ids), None, ids != 0
            materialize(name, count, total, int(config.get('natural_batch_size', 8)), natural_batch)
            del chosen, sequences, tokens
        def grammar_batch(start, stop):
            encoded = tokenizer([row['text'] for row in rows[start:stop]], add_special_tokens=False,
                                padding=True, return_tensors='pt').to(work.device)
            positions = (encoded.attention_mask.sum(1)-1).cpu().numpy()
            return encoded.input_ids, encoded.attention_mask, positions, None
        materialize('grammar_states', len(rows), len(rows), int(config.get('grammar_batch_size', 64)), grammar_batch)
        atomic_json(manifest_path, manifest)
        work.checks.update(actual_final_block_hidden=True, final_layer_norm_oracle=oracle_done,
                           token_zero_states_omitted=True, natural_attention_includes_eos=True,
                           input_order_preserved=True, all_three_caches=True)
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
