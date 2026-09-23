from pathlib import Path
import argparse
import json
import platform
import sys
import traceback

import numpy as np
import torch
import transformers

from run_causalgym_multisite import MultisiteWork, write
from run_shift_request_grid import GridEvaluator, request_id
from train_shift_dictionaries import site_module
from ccad.artifacts import sha256


def prepare(args):
    run = args.reference_run.resolve()
    config = json.loads((run/'config.resolved.json').read_text())
    index = json.loads((run/'request_index.json').read_text())
    models = json.loads(args.models.read_text())
    fitted = models['targets'][str(config['target_seed'])]
    queries = fitted['global_choices']['M_additive']['requests']
    assert queries == [[1.,0.,.5],[0.,1.,0.],[0.,.5,.5]]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    relation_path = run/'native_relation.npz'
    arrays, sites = {}, []
    with np.load(relation_path) as relation:
        for key in relation.files:
            if key.endswith('__support'):
                site = key.removesuffix('__support')
                sites.append(site)
                arrays[key] = relation[key]
                arrays[site+'__coefficients'] = relation[site+'__coefficients']
    probe_path = Path(config['frozen_source_run'])/'probe.npz'
    with np.load(probe_path) as probe:
        arrays.update(probe_weight=probe['weight'].reshape(-1), probe_bias=probe['bias'].reshape(-1))
    arrays['requests'] = np.array(queries, np.float32)
    bank_path = output/'fixed_native.npz'
    np.savez_compressed(bank_path, **arrays)
    targets = {site:dict(path=str(Path(config['target_directory'])/f'{site}_seed{config["target_seed"]}.pt'),
        sha256=index['identity']['target_checkpoint_hashes'][site]) for site in sites}
    manifest = dict(method='M_additive_fixed', target_seed=config['target_seed'], sites=sites,
        goals=['P','N','W'], requests=queries, bank=dict(path=str(bank_path),sha256=sha256(bank_path)),
        target_checkpoints=targets, model_local_dir=config['model_local_dir'], model_revision=config['model_revision'],
        provenance=dict(native_relation_sha256=index['identity']['relation_sha256'],
            exported_coefficients_path=str(relation_path),exported_coefficients_sha256=sha256(relation_path),
            calibration_run=fitted['run'],calibration_documents=fitted['calibration_documents'],
            frozen_models_path=str(args.models.resolve()),frozen_models_sha256=sha256(args.models),
            fixed_head_sha256=sha256(probe_path)),
        execution='Target natural codes at 11 sites; fractions min(1, coefficients @ q); incoming residual preserved',
        precision=dict(dtype='float32',matmul_precision='highest',attention='eager'),
        runtime_inputs='This manifest, fixed_native.npz, target checkpoints, base LM, tokenized panel. Provenance references are not opened during execution.')
    write(output/'deployment.json',manifest)
    panel = json.loads((run/'panel.json').read_text())
    rows = panel['rows']
    indices = sorted(range(len(rows)),key=lambda i:len(rows[i]['tokens']))[:config['eval_batch_size']]
    while len(indices)>1 and len(indices)*max(len(rows[i]['tokens']) for i in indices)>config['token_budget']:
        indices = indices[:-1]
    write(output/'validation_panel.json',dict(rows=[rows[i] for i in indices],
        original_panel=str(run/'panel.json'),original_indices=indices,
        scope='Exact first complete original confirmation batch; deployment equivalence verification'))
    keys = ['device','dictionary_source_dir','dictionary_overlay_dir','eval_batch_size','token_budget',
        'maximum_cuda_bytes','dataset_revision','model_revision','candidate_family_frozen','audit_opened']
    validation = {key:config[key] for key in keys}
    validation.update(run_id=f'CC23_SHIFT_FIXED_NATIVE_T{config["target_seed"]}_20260923',
        run_parent='REUSABLE_STRUCTURE_20260923',run_storage_root=config['run_storage_root'],
        generator_script='scripts/run_shift_fixed_native.py',target_seed=config['target_seed'],seeds=[config['target_seed']],
        purpose='Verify the frozen target-only deployment against its original full-batch grid responses',
        scope='Exact first complete confirmation batch; three frozen target-only requests compared with saved grid responses',
        evidence_level='deployment_equivalence_verification',statistics_unit='Same documents and three fixed requests',
        budget='Existing reusable structure unit; target-only capability delivery, 60 driver seconds and 4GiB CUDA',
        budget_seconds=60,deployment=str(output/'deployment.json'),panel=str(output/'validation_panel.json'),
        verify_reference_run=str(run),replay_atol=1e-5)
    assert not args.config_output.exists()
    write(args.config_output,validation)
    print(json.dumps(dict(deployment=str(output/'deployment.json'),config=str(args.config_output),
        documents=len(indices),sequence_forwards=len(indices)*3,bank_bytes=bank_path.stat().st_size)),flush=True)


class FixedNativeEvaluator(GridEvaluator):
    def __init__(self,work,config):
        self.work, self.config, self.hooks = work, config, []
        self.mode, self.q, self.mask, self.pooled = 'target', None, None, None
        self.reference = None
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        work.torch, work.device = torch, torch.device(config['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable,python_version=platform.python_version(),
            torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,
            cpu_threads=2,matmul_precision='highest',source_assets_loaded=False)
        deployment = json.loads(work.checked(config['deployment']).read_text())
        assert deployment['target_seed'] == config['target_seed']
        assert deployment['model_revision'] == config['model_revision']
        bank_path = work.checked(deployment['bank']['path'])
        assert sha256(bank_path) == deployment['bank']['sha256']
        sys.path.extend([config['dictionary_source_dir'],config['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        self.targets, self.supports, self.coefficients = {}, {}, {}
        with np.load(bank_path) as bank:
            self.requests = bank['requests'].copy()
            self.weight = bank['probe_weight'].copy()
            self.bias = float(bank['probe_bias'][0])
            for site in deployment['sites']:
                checkpoint = deployment['target_checkpoints'][site]
                path = work.checked(checkpoint['path'])
                assert sha256(path) == checkpoint['sha256']
                state = torch.load(path,map_location=work.device,weights_only=True)
                target = AutoEncoderTopK(512,state['encoder.weight'].shape[0],int(state['k'])).to(work.device)
                target.load_state_dict(state)
                self.targets[site] = target.eval().requires_grad_(False)
                self.supports[site] = torch.tensor(bank[site+'__support'],device=work.device,dtype=torch.long)
                self.coefficients[site] = torch.tensor(bank[site+'__coefficients'],device=work.device,dtype=torch.float32)
        np.testing.assert_array_equal(self.requests,deployment['requests'])
        panel = json.loads(work.checked(config['panel']).read_text())
        self.rows = panel['rows']
        self.documents = np.array([row['document_sha256'] for row in self.rows])
        assert len(set(self.documents)) == len(self.rows)>0
        write(work.run/'panel.json',panel)
        for name in ('config.json','model.safetensors'):
            work.checked(Path(deployment['model_local_dir'])/name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(deployment['model_local_dir'],
            local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval().to(work.device)
        self.model.requires_grad_(False)
        for site in deployment['sites']:
            self.hooks.append(site_module(self.model,site).register_forward_hook(self.hook(site)))


def execute(args):
    config = json.loads(args.config.read_text())
    work = MultisiteWork(config,args.config,['scripts/run_shift_fixed_native.py','scripts/run_shift_request_grid.py',
        'scripts/train_shift_dictionaries.py','scripts/run_causalgym_multisite.py','src/ccad/artifacts.py'])
    evaluator, error = None, None
    try:
        evaluator = FixedNativeEvaluator(work,config)
        references = {}
        if config.get('verify_reference_run'):
            index = json.loads(work.checked(Path(config['verify_reference_run'])/'request_index.json').read_text())
            references = {row['request_id']:row for row in index['target_blocks']}
        records = []
        for goal,q in zip(('P','N','W'),evaluator.requests):
            values,_ = evaluator.evaluate('target',q)
            logits = values@evaluator.weight+evaluator.bias
            path = work.run/f'{goal}.npz'
            np.savez_compressed(path,pooled512=values,logits=logits,request=q,document_sha256=evaluator.documents)
            record = dict(goal=goal,request=q.tolist(),path=str(path),documents=len(values))
            if references:
                with np.load(work.checked(references[request_id(q)]['path'])) as old:
                    lookup = {str(value):i for i,value in enumerate(old['document_sha256'])}
                    indices = [lookup[str(value)] for value in evaluator.documents]
                    record['pooled_max_absolute_error'] = float(np.max(np.abs(values-old['pooled512'][indices])))
                    record['logit_max_absolute_error'] = float(np.max(np.abs(logits-old['logits'][indices])))
                    np.testing.assert_allclose(values,old['pooled512'][indices],atol=config['replay_atol'],rtol=0)
                    np.testing.assert_allclose(logits,old['logits'][indices],atol=config['replay_atol'],rtol=0)
            records.append(record)
            write(work.run/'deployment_results.json',dict(method='M_additive_fixed',records=records))
            work.record(kind='fixed_native',task='deployment',row_id=goal,component=goal,method='M_additive_fixed',
                operation=goal,seed=config['target_seed'],target_seed=config['target_seed'],**record)
            work.progress('FIXED_NATIVE',completed_requests=len(records),total_requests=3)
        work.checks.update(three_fixed_requests_saved=len(records)==3,target_only_initialization=True,
            frozen_target=True,grid_replay_verified=bool(references) or 'not_requested')
    except Exception:
        error = traceback.format_exc()
    finally:
        if evaluator is not None:
            evaluator.close()
    status = work.finish(error)
    if error:
        raise RuntimeError(error)
    return status


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command',required=True)
    preparation = commands.add_parser('prepare')
    preparation.add_argument('--reference-run',type=Path,required=True)
    preparation.add_argument('--models',type=Path,required=True)
    preparation.add_argument('--output',type=Path,required=True)
    preparation.add_argument('--config-output',type=Path,required=True)
    execution = commands.add_parser('run')
    execution.add_argument('--config',type=Path,required=True)
    args = parser.parse_args()
    return prepare(args) if args.command=='prepare' else execute(args)


if __name__=='__main__':
    raise SystemExit(main())
