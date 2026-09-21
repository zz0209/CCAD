from pathlib import Path
import argparse
import hashlib
import io
import json
import re
import sys
import traceback
import zipfile
import numpy as np
import torch
import transformers
from run_causalgym_multisite import MultisiteWork, write
from train_shift_dictionaries import site_module
from adaptive_native_execution import realize


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    config = json.loads((ROOT/settings['base_config']).read_text()) | settings
    work = MultisiteWork(config, args.config, ['scripts/check_number_program.py',
        'scripts/train_shift_dictionaries.py', 'scripts/run_causalgym_multisite.py',
        'scripts/run_r011s1_raw_hook_asset.py', 'scripts/adaptive_native_execution.py', 'src/ccad/artifacts.py'])
    error = None
    handles = []
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        work.device = torch.device(config['device'])
        work.torch = torch
        if work.device.type == 'cuda':
            torch.cuda.set_device(work.device)
            torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, torch=torch.__version__,
            transformers=transformers.__version__, numpy=np.__version__, device=str(work.device), threads=2)
        work.checked(ROOT/settings['base_config'])
        source_path = work.checked(ROOT/config['annotations'], 'SFC human annotations', 'MIT')
        annotations = [json.loads(line) for line in source_path.read_text().splitlines()]
        selection = {}
        for annotation in annotations:
            name = annotation['Name']
            if '/' not in name:
                continue
            site, member = name.split('/')
            labels = re.findall(r'\b(singular|plural)\b', annotation['Annotation'].lower())
            labels = sorted(set(labels))
            if site not in config['train_sites'] or len(labels) != 1:
                continue
            record = dict(member=int(member), group=labels[0], annotation=annotation)
            previous = selection.setdefault(site, {}).get(int(member))
            if previous is not None and previous != record:
                raise ValueError(f'Conflicting annotation for {name}')
            selection[site][int(member)] = record
        selection = {site: sorted(selection[site].values(), key=lambda r:r['member'])
                     for site in config['train_sites'] if site in selection}
        rows = json.loads(work.checked(ROOT/config['number_panel']).read_text())['rows']
        if config['per_structure']:
            rows = [r for structure in ['simple', 'within_rc', 'rc']
                    for r in [x for x in rows if x['structure'] == structure][:config['per_structure']]]
        if not rows or not selection:
            raise ValueError('Empty source program or context panel')
        rows = [dict(row, target_number=row['case'].split('_')[int(row['structure'] == 'within_rc')])
                for row in rows]
        if any(row['target_number'] not in ['singular', 'plural'] for row in rows):
            raise ValueError('Unknown target number')
        write(work.run/'source_design.json', dict(selection=selection, rows=rows,
            rule='All annotated singular or plural members at the existing target sites, with exactly one number label.',
            hypothesis='Deleting the matching number part changes the clean-versus-patch answer margin more than deleting the opposite part.',
            scope='Public annotations define this source program before target-response measurement. The selected program uses available sites and is distinct from the complete SFC circuit.',
            target_dictionaries_used=bool(config.get('target_methods')),
            target_responses_used_for_selection=False))
        archive = Path(config['source_archive'])
        if not archive.is_file():
            raise FileNotFoundError(archive)
        bank = {}
        identities = {}
        with zipfile.ZipFile(archive) as stream:
            for site, members in selection.items():
                component = 'embed' if site == 'embed' else site.replace('_', '_out_layer')
                location = f'dictionaries/pythia-70m-deduped/{component}/10_32768/ae.pt'
                data = stream.read(location)
                state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
                ids = [r['member'] for r in members]
                bank[site] = dict(encoder=state['encoder.weight'][ids].clone(),
                    encoder_bias=state['encoder.bias'][ids].clone(),
                    decoder=state['decoder.weight'][:, ids].T.clone(), center=state['bias'].clone())
                identities[location] = hashlib.sha256(data).hexdigest()
                work.progress('SOURCE_MEMBERS', site=site, members=len(ids))
                del state, data
        write(work.run/'source_identity.json', dict(archive=str(archive), members=identities))
        np.savez(work.run/'source_parameters.npz', **{site+'__'+key:value.numpy()
                  for site, tensors in bank.items() for key, value in tensors.items()})
        bank = {site:{key:value.to(work.device) for key,value in tensors.items()}
                for site,tensors in bank.items()}
        target_methods = config.get('target_methods', [])
        targets = {}
        if target_methods:
            sys.path.extend([config['dictionary_source_dir'], config['dictionary_overlay_dir']])
            from dictionary_learning.trainers.top_k import AutoEncoderTopK
            work.checked(Path(config['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py', 'Pinned TopK', 'MIT')
            for site in selection:
                path = Path(config['target_directory'])/f'{site}_seed{config["target_seed"]}.pt'
                state = torch.load(work.checked(path), map_location='cpu', weights_only=True)
                target = AutoEncoderTopK(512, len(state['encoder.weight']), int(state['k'])).to(work.device)
                target.load_state_dict(state)
                target.requires_grad_(False)
                targets[site] = target
        model = transformers.AutoModelForCausalLM.from_pretrained(config['model_local_dir'],
            local_files_only=True, dtype=torch.float32, attn_implementation='eager').to(work.device).eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        tokenizer = transformers.AutoTokenizer.from_pretrained(config['model_local_dir'], local_files_only=True)
        for filename in ['config.json', 'model.safetensors', 'tokenizer.json']:
            work.checked(Path(config['model_local_dir'])/filename, 'Pinned Pythia70M', 'Apache-2.0')
        current = 'none'
        execution = 'source'
        trajectory = {}
        source_actions = {}
        execution_counts = {name:dict(states=0, changed=0, min_code=0., max_members=0)
                            for name in target_methods if name != 'raw_readout'}
        queries = {name:{site:torch.tensor([float(name == 'full' or r['group'] == name)
                   for r in members], device=work.device) for site, members in selection.items()}
                   for name in ['singular', 'plural', 'full']}
        def make_hook(site):
            def apply(module, inputs, output):
                if current == 'none':
                    return output
                value = output[0] if isinstance(output, tuple) else output
                source = bank[site]
                code = torch.relu((value-source['center'])@source['encoder'].T+source['encoder_bias'])
                action = -(code*queries[current][site])@source['decoder']
                if execution == 'source':
                    value = value+action
                    trajectory[site] = value.detach().clone()
                    source_actions[site] = action.detach().clone()
                elif execution == 'raw_readout':
                    reconstructed = targets[site].decode(targets[site].encode(value))
                    readout = torch.relu((reconstructed-source['center'])@source['encoder'].T+source['encoder_bias'])
                    value = value-(readout*queries[current][site])@source['decoder']
                else:
                    if execution == 'local_action':
                        desired = action
                    elif execution == 'recorded_action':
                        desired = source_actions[site]
                    elif execution == 'state_feedback':
                        desired = trajectory[site]-value
                    else:
                        raise ValueError(execution)
                    target = targets[site]
                    codes = target.encode(value)
                    candidate = target.encode(value+desired)-codes
                    allowance = 2*len(selection[site])
                    delta, coefficients, indices, detail = realize(desired, candidate,
                        target.decoder.weight.T, members=allowance, steps=config['inverse_steps'],
                        refine_steps=config['inverse_steps'], batch_size=128,
                        candidate_limit=config['inverse_candidates'], current_codes=codes)
                    remaining = torch.gather(codes, -1, indices)+coefficients
                    count = execution_counts[execution]
                    count['states'] += codes.numel()//codes.shape[-1]
                    count['changed'] += int((coefficients != 0).sum())
                    count['min_code'] = min(count['min_code'], float(remaining.min()))
                    count['max_members'] = max(count['max_members'], int((coefficients != 0).sum(-1).max()))
                    if float(remaining.min()) < -2e-5 or int((coefficients != 0).sum(-1).max()) > allowance:
                        raise ValueError('Invalid target execution')
                    value = value+delta
                return (value, *output[1:]) if isinstance(output, tuple) else value
            return apply
        for site in selection:
            handles.append(site_module(model, site).register_forward_hook(make_hook(site)))
        predictions = {name:[] for name in ['none', *queries]}
        predictions.update({method+'__'+name:[] for method in target_methods for name in queries})
        with torch.inference_mode():
            for index, row in enumerate(rows):
                encoded = tokenizer(row['clean_prefix'], return_tensors='pt', add_special_tokens=False).to(work.device)
                answers = [tokenizer.encode(row[key], add_special_tokens=False)
                           for key in ['clean_answer', 'patch_answer']]
                if any(len(answer) != 1 for answer in answers):
                    raise ValueError('Expected two single-token answers')
                for current in ['none', *queries]:
                    execution = 'source'
                    output = model(**encoded).logits[0, -1]
                    predictions[current].append(float(output[answers[0][0]]-output[answers[1][0]]))
                    work.sequence_forwards += 1
                    work.token_forwards += encoded['input_ids'].numel()
                    if current != 'none':
                        for execution in target_methods:
                            output = model(**encoded).logits[0, -1]
                            predictions[execution+'__'+current].append(float(output[answers[0][0]]-output[answers[1][0]]))
                            work.sequence_forwards += 1
                            work.token_forwards += encoded['input_ids'].numel()
                np.savez(work.run/'responses.npz', **{k:np.asarray(v) for k,v in predictions.items()})
                work.progress('SOURCE_EVALUATION', completed=index+1, total=len(rows))
        clean = np.asarray(predictions['none'])
        statistics = {}
        for structure in [*sorted({row['structure'] for row in rows}), 'all']:
            chosen = np.array([structure == 'all' or r['structure'] == structure for r in rows])
            matched = np.asarray([predictions[row['target_number']][i] for i,row in enumerate(rows)])
            other = np.asarray([predictions['plural' if row['target_number'] == 'singular' else 'singular'][i]
                                for i,row in enumerate(rows)])
            statistics[structure] = dict(documents=int(chosen.sum()),
                accuracy={k:float((np.asarray(v)[chosen] > 0).mean()) for k,v in predictions.items() if '__' not in k},
                mean_effect={k:float((np.asarray(v)[chosen]-clean[chosen]).mean()) for k,v in predictions.items() if '__' not in k},
                matched_margin_change=float((matched-clean)[chosen].mean()),
                opposite_margin_change=float((other-clean)[chosen].mean()),
                matched_deletion_larger_fraction=float((matched[chosen] < other[chosen]).mean()))
        write(work.run/'source_analysis.json', statistics)
        if target_methods:
            target_results = {}
            for method in target_methods:
                target_results[method] = {}
                for name in queries:
                    reference = np.asarray(predictions[name])
                    prediction = np.asarray(predictions[method+'__'+name])
                    denominator = np.mean((reference-clean)**2)
                    if denominator <= 1e-12:
                        raise ValueError('No source effect for target evaluation')
                    target_results[method][name] = float(np.sqrt(np.mean((prediction-reference)**2)/denominator))
            write(work.run/'target_analysis.json', dict(metrics=target_results, execution=execution_counts,
                scope=config['scope']))
        work.checks.update(nonempty=True, fixed_source=True, finite=all(np.isfinite(v).all() for v in predictions.values()))
        if not work.checks['finite']:
            raise ValueError('Nonfinite source response')
        for index, (structure, record) in enumerate(statistics.items()):
            work.record(kind='source_number', task=structure, row_id=index, component=structure, **record)
    except BaseException:
        error = traceback.format_exc()
        raise
    finally:
        for handle in handles:
            handle.remove()
        code = work.finish(error)
        if error is None and code:
            raise RuntimeError('Source run validation failed')


if __name__ == '__main__':
    main()
