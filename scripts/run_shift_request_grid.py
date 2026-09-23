from pathlib import Path
import argparse
import itertools
import json
import platform
import sys
import time
import traceback

import numpy as np
import torch
import transformers

from run_causalgym_multisite import MultisiteWork, write
from run_shift_explanation import source_groups
from train_shift_dictionaries import site_module
from train_shift_finite_parts import load_config
from ccad.artifacts import sha256


def request_id(q):
    return 'q_'+''.join(str(int(round(2*value))) for value in q)


class GridEvaluator:
    def __init__(self, work, config):
        self.work, self.config = work, config
        self.hooks = []
        self.mode, self.q, self.mask, self.pooled = 'source', None, None, None
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        work.torch, work.device = torch, torch.device(config['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
            cpu_threads=2, matmul_precision='highest')
        sys.path.extend([config['dictionary_source_dir'], config['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        source_path = work.checked(config['source_manifest'])
        source = json.loads(source_path.read_text())
        notebook_path = work.checked(config['notebook'])
        groups,_ = source_groups(notebook_path, source['members'])
        parameter_path = work.checked(config['source_parameters'])
        bank = np.load(parameter_path)
        relation_path = work.checked(Path(config['relation_run'])/'relation.npz')
        relation = np.load(relation_path)
        self.parameters, self.partitions, self.targets, self.supports, self.coefficients = {}, {}, {}, {}, {}
        target_hashes, exported = {}, {}
        for site, members in source['members'].items():
            self.parameters[site] = {key: torch.tensor(bank[site+'__'+key],device=work.device)
                for key in ('encoder','encoder_bias','decoder','center')}
            partition = np.array([[member in groups[name].get(site,[]) for name in
                ('pronouns','names','associated_words')] for member in members],dtype=np.float32)
            assert np.all(partition.sum(1)==1)
            self.partitions[site] = torch.tensor(partition,device=work.device)
            native = relation[site+'__native']
            support = np.flatnonzero(native.sum(1)>0)
            coefficients = native[support]@partition
            assert coefficients.min() >= 0 and coefficients.sum(1).max() <= 1.000001
            self.supports[site] = torch.tensor(support,device=work.device,dtype=torch.long)
            self.coefficients[site] = torch.tensor(coefficients,device=work.device,dtype=torch.float32)
            path = work.checked(Path(config['target_directory'])/f'{site}_seed{config["target_seed"]}.pt')
            state = torch.load(path,map_location=work.device,weights_only=True)
            target = AutoEncoderTopK(512,state['encoder.weight'].shape[0],int(state['k'])).to(work.device)
            target.load_state_dict(state)
            self.targets[site] = target.eval().requires_grad_(False)
            target_hashes[site] = sha256(path)
            exported.update({site+'__support':support,site+'__coefficients':coefficients,site+'__partition':partition})
        np.savez_compressed(work.run/'native_relation.npz',**exported)
        probe_path = work.checked(Path(config['frozen_source_run'])/'probe.npz')
        with np.load(probe_path) as probe:
            self.weight = probe['weight'].reshape(-1)
            self.bias = float(probe['bias'].reshape(-1)[0])
        panel_path = work.checked(config['panel'])
        panel = json.loads(panel_path.read_text())
        self.rows = panel['rows']
        self.documents = np.array([row['document_sha256'] for row in self.rows])
        assert len(set(self.documents)) == len(self.rows) and len(self.rows)>0
        write(work.run/'panel.json',panel)
        self.source_identity = dict(documents=self.documents.tolist(),source_parameters_sha256=sha256(parameter_path),
            source_members_sha256=sha256(source_path),notebook_sha256=sha256(notebook_path),probe_sha256=sha256(probe_path),
            model_revision=config['model_revision'],matmul_precision='highest',parts=['P','N','W'])
        self.identity = dict(source=self.source_identity,target_seed=config['target_seed'],target_checkpoint_hashes=target_hashes,
            relation_sha256=sha256(relation_path),phase=config['phase'])
        self.reference = None
        if config.get('boolean_reference_file'):
            reference_path = work.checked(config['boolean_reference_file'])
            with np.load(reference_path) as reference:
                document_lookup = {str(value):i for i,value in enumerate(reference['document_sha256'])}
                row_indices = [document_lookup[str(value)] for value in self.documents]
                self.reference = dict(pooled=reference['pooled512'][row_indices].copy(),
                    methods=reference['method_names'].tolist(),requests=reference['requests'].copy())
        for name in ('config.json','model.safetensors'):
            work.checked(Path(config['model_local_dir'])/name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(config['model_local_dir'],local_files_only=True,
            dtype=torch.float32,attn_implementation='eager').eval().to(work.device)
        self.model.requires_grad_(False)
        for site in self.parameters:
            self.hooks.append(site_module(self.model,site).register_forward_hook(self.hook(site)))

    def hook(self, site):
        def apply(module,inputs,output):
            x = output[0] if isinstance(output,tuple) else output
            if self.mode == 'source':
                parameters = self.parameters[site]
                codes = torch.relu((x-parameters['center'])@parameters['encoder'].T+parameters['encoder_bias'])
                x = x-(codes*(self.partitions[site]@self.q))@parameters['decoder']
            else:
                codes = self.targets[site].encode(x)[...,self.supports[site]]
                fraction = (self.coefficients[site]@self.q).clamp(max=1.)
                x = x-(codes*fraction)@self.targets[site].decoder.weight[:,self.supports[site]].T
            if site == 'resid_4':
                self.pooled = (x*self.mask[...,None]).sum(1)/self.mask.sum(1)[:,None]
            return (x,*output[1:]) if isinstance(output,tuple) else x
        return apply

    @torch.no_grad()
    def evaluate(self, family, q):
        self.mode, self.q = family, torch.tensor(q,device=self.work.device,dtype=torch.float32)
        values = np.empty((len(self.rows),512),np.float32)
        order = sorted(range(len(self.rows)),key=lambda i:len(self.rows[i]['tokens']))
        offset = 0
        while offset<len(order):
            indices = order[offset:offset+self.config.get('eval_batch_size',8)]
            while len(indices)>1 and len(indices)*max(len(self.rows[i]['tokens']) for i in indices)>self.config['token_budget']:
                indices = indices[:-1]
            width = max(len(self.rows[i]['tokens']) for i in indices)
            assert len(indices)*width <= self.config['token_budget']
            ids = torch.zeros((len(indices),width),device=self.work.device,dtype=torch.long)
            self.mask = torch.zeros_like(ids)
            for j,index in enumerate(indices):
                tokens = self.rows[index]['tokens']
                ids[j,:len(tokens)] = torch.tensor(tokens,device=self.work.device)
                self.mask[j,:len(tokens)] = 1
            if time.perf_counter()-self.work.wall_start > self.config['budget_seconds']:
                raise TimeoutError('Request-grid driver budget exceeded')
            self.model.gpt_neox(ids,attention_mask=self.mask,use_cache=False)
            self.work.sequence_forwards += len(indices)
            self.work.token_forwards += ids.numel()
            values[indices] = self.pooled.cpu().numpy()
            assert torch.cuda.max_memory_allocated(self.work.device) <= self.config.get('maximum_cuda_bytes',4*1024**3)
            offset += len(indices)
        assert np.isfinite(values).all()
        replay_error = None
        if self.reference is not None and all(value in (0.,1.) for value in q):
            qi = np.flatnonzero(np.all(self.reference['requests']==np.array(q),axis=1)).item()
            mi = self.reference['methods'].index('source' if family=='source' else 'native')
            expected = self.reference['pooled'][:,mi,qi]
            replay_error = float(np.max(np.abs(values-expected)))
            np.testing.assert_allclose(values,expected,atol=self.config.get('replay_atol',1e-5),rtol=0)
        return values,replay_error

    def close(self):
        for handle in self.hooks:
            handle.remove()


def collect(work,evaluator,config):
    grid = list(itertools.product((0.,.5,1.),repeat=3))
    assert config['phase'] in ('calibration','evaluation')
    queries = dict(source=[q for q in grid if all(value in (0.,1.) for value in q)] if config['phase']=='calibration' else grid,
        target=[(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)] if config['phase']=='calibration' else grid)
    index = dict(identity=evaluator.identity,source_identity=evaluator.source_identity,phase=config['phase'],
        source_blocks=[],target_blocks=[],conditions='Dynamic same-hook P/N/W natural fractions; pooled before final layernorm',
        deployment_source_requests='Only Boolean8; fractional source requests are diagnostic observations')
    cached = {}
    for key in ('source_run','resume_run'):
        if not config.get(key):
            continue
        old_path = work.checked(Path(config[key])/'request_index.json')
        old = json.loads(old_path.read_text())
        assert old['source_identity'] == evaluator.source_identity
        families = ['source'] if key=='source_run' else ['source','target']
        if key=='resume_run':
            assert old['identity'] == evaluator.identity
        for family in families:
            for record in old[family+'_blocks']:
                cached[family,record['request_id']] = record
    directory = work.run/'response_blocks'
    directory.mkdir()
    write(work.run/'request_index.json',index)
    for family in ('source','target'):
        for number,q in enumerate(queries[family]):
            name = request_id(q)
            reused = (family,name) in cached
            if reused:
                record = dict(cached[family,name])
                with np.load(work.checked(record['path'])) as block:
                    assert str(block['status'])=='PASS'
                    np.testing.assert_array_equal(block['document_sha256'],evaluator.documents)
                    np.testing.assert_array_equal(block['request'],q)
                    values = block['pooled512']
                    assert np.isfinite(values).all()
                record['reused_from'] = record['path']
            else:
                start = time.perf_counter()
                sequences,tokens = work.sequence_forwards,work.token_forwards
                values,replay_error = evaluator.evaluate(family,q)
                path = directory/f'{family}_{name}.npz'
                np.savez_compressed(path,pooled512=values,logits=values@evaluator.weight+evaluator.bias,
                    request=np.array(q),document_sha256=evaluator.documents,status=np.array('PASS'),family=np.array(family))
                record = dict(request_id=name,request=list(q),path=str(path.resolve()),status='PASS',
                    sequence_forwards=work.sequence_forwards-sequences,token_forwards=work.token_forwards-tokens,
                    wall_seconds=time.perf_counter()-start,replay_maximum_absolute_error=replay_error)
            index[family+'_blocks'].append(record)
            write(work.run/'request_index.json',index)
            work.record(kind='request_grid',task=config['phase'],row_id=number,component=name,method=family,
                operation=name,seed=config['target_seed'],target_seed=config['target_seed'],split=config['phase'],
                documents=len(evaluator.rows),reused=reused,pooled_mean_square=float(np.mean(values.astype(np.float64)**2)),path=record['path'])
            work.progress('REQUEST_GRID',phase=config['phase'],family=family,completed_requests=number+1,
                total_requests=len(queries[family]),documents=len(evaluator.rows),reused=reused)
    work.checks.update(all_requested_blocks_saved=True,finite_pooled=True,frozen_original_native=True,
        dynamic_site_reencoding=True,source_boolean_and_fractional_information_separated=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    work = MultisiteWork(config,args.config,['scripts/run_shift_request_grid.py','scripts/train_shift_finite_parts.py',
        'scripts/run_shift_explanation.py','scripts/train_shift_dictionaries.py','scripts/run_causalgym_multisite.py','src/ccad/artifacts.py'])
    evaluator,error = None,None
    try:
        evaluator = GridEvaluator(work,config)
        collect(work,evaluator,config)
    except Exception:
        error = traceback.format_exc()
    finally:
        if evaluator is not None:
            evaluator.close()
    status = work.finish(error)
    if error:
        raise RuntimeError(error)
    return status


if __name__=='__main__':
    raise SystemExit(main())
