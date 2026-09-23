from pathlib import Path
import argparse
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
from run_shift_member_responses import identity_hash
from train_shift_dictionaries import site_module
from ccad.artifacts import sha256


def load_config(path):
    current = json.loads(Path(path).read_text())
    return {**load_config(current['base_config']), **current} if current.get('base_config') else current


class GroupEvaluator:
    def __init__(self, work, config):
        self.work, self.config = work, config
        self.hooks = []
        self.condition, self.active, self.mask, self.pooled = 0, {}, None, None
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        work.torch, work.device = torch, torch.device(config['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        sys.path.extend([config['dictionary_source_dir'], config['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        work.environment = dict(python=sys.executable, python_version=platform.python_version(),
            torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
            matmul_precision='highest', cpu_threads=2, device=str(work.device))
        calibration_run = Path(config['calibration_run'])
        assert json.loads((calibration_run/'status.json').read_text())['status'] == 'PASS'
        index = json.loads(work.checked(calibration_run/'response_index.json').read_text())
        old_config = json.loads(work.checked(calibration_run/'config.resolved.json').read_text())
        assert old_config['target_seed'] == config['target_seed']
        for key in ('source_manifest', 'source_parameters', 'relation_run', 'target_directory', 'model_local_dir'):
            assert Path(config[key]).resolve() == Path(old_config[key]).resolve(), key
        panel = json.loads(work.checked(config['evaluation_panel']).read_text())
        self.rows = panel['rows']
        self.documents = np.array([row['document_sha256'] for row in self.rows])
        with np.load(work.checked(index['source_reference'], 'Shared eight-document source reference')) as reference:
            assert str(reference['status']) == 'PASS'
            np.testing.assert_array_equal(reference['document_sha256'], self.documents)
            self.baseline = reference['baseline_pooled512'].copy()
            self.teacher = reference['whole_W_pooled512'].copy()
        assert self.teacher.shape == self.baseline.shape == (8, 2, 512)
        self.candidates = {site: np.array(ids, dtype=np.int64) for site, ids in index['candidate_ids'].items()}
        self.quotas = config['members_by_site']
        self.sites = list(self.quotas)
        assert set(self.candidates) == set(self.sites) and sum(self.quotas.values()) == 22
        self.candidate_count = sum(map(len, self.candidates.values()))
        source = json.loads(work.checked(config['source_manifest'], 'Published source members', 'MIT').read_text())
        self.sites = [site for site in source['members'] if site in self.quotas]
        groups, _ = source_groups(work.checked(config['notebook'], 'Published source annotations', 'MIT'), source['members'])
        bank = np.load(work.checked(config['source_parameters'], 'Frozen source parameters', 'MIT'))
        self.parameters, self.targets, target_hashes = {}, {}, {}
        self.initial_probability, self.native_support, self.native_mass = {}, {}, {}
        relation = np.load(work.checked(Path(config['relation_run'])/'relation.npz'))
        for site, members in source['members'].items():
            params = {key: torch.tensor(bank[site+'__'+key], device=work.device)
                      for key in ('encoder', 'encoder_bias', 'decoder', 'center')}
            params['p'] = torch.tensor([i in groups['pronouns'].get(site, []) for i in members], device=work.device)
            self.parameters[site] = params
            if site in self.sites:
                np.testing.assert_array_equal(relation[site+'__candidates'], self.candidates[site])
                word_mask = np.array([member in groups['associated_words'].get(site, []) for member in members])
                mass = relation[site+'__native'].astype(np.float64)[:, word_mask].sum(1)[self.candidates[site]]
                assert np.isfinite(mass).all() and np.all(mass >= 0) and mass.sum() > 0, f'Invalid native W mass at {site}'
                self.native_mass[site] = mass.tolist()
                self.initial_probability[site] = .5*mass/mass.sum()+.5/len(mass)
                self.native_support[site] = self.candidates[site][np.argsort(-mass, kind='stable')[:self.quotas[site]]].astype(int).tolist()
                path = work.checked(Path(config['target_directory'])/f'{site}_seed{config["target_seed"]}.pt')
                state = torch.load(path, map_location=work.device, weights_only=True)
                target = AutoEncoderTopK(512, state['encoder.weight'].shape[0], int(state['k'])).to(work.device)
                target.load_state_dict(state)
                self.targets[site] = target.eval().requires_grad_(False)
                target_hashes[site] = sha256(path)
        for filename in ('config.json', 'model.safetensors'):
            work.checked(Path(config['model_local_dir'])/filename, 'Pinned Pythia70M', 'Apache-2.0')
        self.model = transformers.AutoModelForCausalLM.from_pretrained(config['model_local_dir'],
            local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval().to(work.device)
        self.model.requires_grad_(False)
        for site in source['members']:
            self.hooks.append(site_module(self.model, site).register_forward_hook(self.hook(site)))
        self.identity = dict(documents=self.documents.tolist(), source_reference_sha256=sha256(Path(index['source_reference'])),
            target_checkpoint_hashes=target_hashes, candidate_ids=index['candidate_ids'], quotas=self.quotas,
            conditions=['none', 'dynamic_source_P'], model_revision=config['model_revision'],
            source_parameters_sha256=sha256(Path(config['source_parameters'])),
            operation='Complete natural target member deletion; source P and target W read the same incoming state at each site')
        write(work.run/'evaluation_identity.json', self.identity)
        write(work.run/'panel.json', panel)
        np.savez_compressed(work.run/'source_reference.npz', document_sha256=self.documents,
            baseline_pooled512=self.baseline, whole_W_pooled512=self.teacher)

    def hook(self, site):
        def apply(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            delta = torch.zeros_like(x)
            if self.condition:
                source = self.parameters[site]
                codes = torch.relu((x-source['center'])@source['encoder'].T+source['encoder_bias'])
                delta = -(codes*source['p'])@source['decoder']
            if site in self.active:
                members = self.active[site]
                codes = self.targets[site].encode(x)
                selected = codes.gather(2, members[:, None, :].expand(-1, x.shape[1], -1))
                decoder = self.targets[site].decoder.weight.T[members]
                delta = delta-torch.bmm(selected, decoder)
            updated = x+delta
            if site == 'resid_4':
                self.pooled = (updated*self.mask[..., None]).sum(1)/self.mask.sum(1)[:, None]
            return (updated, *output[1:]) if isinstance(output, tuple) else updated
        return apply

    @torch.no_grad()
    def evaluate(self, groups):
        for group in groups:
            assert set(group) == set(self.sites)
            for site in self.sites:
                assert len(group[site]) == len(set(group[site])) == self.quotas[site]
                assert set(group[site]) <= set(self.candidates[site].tolist())
        count = len(groups)
        hidden = np.empty((count, len(self.rows), 2, 512), dtype=np.float32)
        order = sorted(range(len(self.rows)), key=lambda i: len(self.rows[i]['tokens']))
        for self.condition in (0, 1):
            offset = 0
            while offset < len(order):
                selected = order[offset:offset+self.config['batch_size']]
                while len(selected)>1 and count*len(selected)*max(len(self.rows[i]['tokens']) for i in selected)>self.config['token_budget']:
                    selected = selected[:-1]
                width = max(len(self.rows[i]['tokens']) for i in selected)
                assert count*len(selected)*width <= self.config['token_budget']
                ids = torch.zeros((len(selected), width), dtype=torch.long, device=self.work.device)
                attention = torch.zeros_like(ids)
                for j, index in enumerate(selected):
                    tokens = self.rows[index]['tokens']
                    ids[j, :len(tokens)] = torch.tensor(tokens, device=self.work.device)
                    attention[j, :len(tokens)] = 1
                ids, self.mask = ids.repeat(count, 1), attention.repeat(count, 1)
                self.active = {site: torch.tensor([g[site] for g in groups], device=self.work.device).repeat_interleave(len(selected), dim=0)
                               for site in self.sites}
                if time.perf_counter()-self.work.wall_start > self.config['budget_seconds']:
                    raise TimeoutError('Complete group search driver budget exceeded')
                self.model.gpt_neox(ids, attention_mask=self.mask, use_cache=False)
                self.work.sequence_forwards += len(ids)
                self.work.token_forwards += ids.numel()
                values = self.pooled.detach().cpu().numpy().reshape(count, len(selected), 512)
                assert np.isfinite(values).all()
                hidden[:, selected, self.condition] = values
                assert torch.cuda.max_memory_allocated(self.work.device) <= self.config.get('maximum_cuda_bytes', 4*1024**3)
                offset += len(selected)
        # 相同背景的baseline相减后抵消，直接比较完整执行与完整source目标。
        errors = hidden.astype(np.float64)-self.teacher[None].astype(np.float64)
        return hidden, np.sum(errors**2, axis=(1, 2, 3)), np.sum(errors**2, axis=(1, 3))

    def close(self):
        for handle in self.hooks:
            handle.remove()


def group_key(group, sites):
    return tuple(tuple(sorted(group[site])) for site in sites)


def sample_group(probability, evaluator, rng):
    result = {}
    for site in evaluator.sites:
        score = np.log(probability[site])+rng.gumbel(size=len(probability[site]))
        order = np.lexsort((evaluator.candidates[site], -score))[:evaluator.quotas[site]]
        result[site] = sorted(evaluator.candidates[site][order].astype(int).tolist())
    return result


def save_state(path, state):
    temporary = path.with_suffix('.tmp')
    write(temporary, state)
    temporary.replace(path)


def best_observation(observations, sites):
    return min(observations, key=lambda row: (row['loss'], group_key(row['members'], sites)))


def run_search(work, evaluator, config):
    search = config['search']
    assert search['population'] == 16 and search['elite'] == 4
    assert search['seed'] == 2026092314+config['target_seed']
    budget = int(search.get('group_budget', evaluator.candidate_count))
    assert 1 <= budget <= evaluator.candidate_count
    protocol = dict(evaluation=evaluator.identity, search=dict(search, group_budget=budget),
        site_order=evaluator.sites, native_support=evaluator.native_support, native_mass=evaluator.native_mass,
        initial_probability={site:value.tolist() for site,value in evaluator.initial_probability.items()},
        objective='Sum of squared differences between actual target and source whole-W pooled512 over eight documents and both backgrounds',
        budget_unit='One distinct complete group costs 8 documents times 2 backgrounds = 16 target sequence forwards',
        native_initial_update=False, snapshot_queries=[32, 64, 128, budget])
    protocol_hash = identity_hash(protocol)
    write(work.run/'search_protocol.json', dict(protocol, sha256=protocol_hash))
    state = dict(protocol_sha256=protocol_hash, methods={})
    if config.get('resume_run'):
        state = json.loads(work.checked(Path(config['resume_run'])/'search_state.json', 'Saved completed group search state').read_text())
        assert state['protocol_sha256'] == protocol_hash
    bank_dir = work.run/'group_bank'
    bank_dir.mkdir()
    state_path = work.run/'search_state.json'
    selections, summaries = {}, {}
    for method in ('adaptive', 'passive'):
        if method not in state['methods']:
            state['methods'][method] = dict(probability=protocol['initial_probability'],
                rng_state=np.random.default_rng(search['seed']).bit_generator.state,
                observations=[], proposals=[], duplicate_proposals=0, population=0,
                pending=[evaluator.native_support], pending_kind='native', snapshots={})
        current = state['methods'][method]
        rng = np.random.default_rng()
        rng.bit_generator.state = current['rng_state']
        for previous in current['observations']:
            with np.load(previous['path']) as block:
                assert str(block['status']) == 'PASS' and str(block['protocol_sha256']) == protocol_hash
                np.testing.assert_array_equal(block['document_sha256'], evaluator.documents)
        save_state(state_path, state)
        while len(current['observations']) < budget or current['pending']:
            if not current['pending']:
                remaining = budget-len(current['observations'])
                if remaining == 0:
                    break
                keys = {group_key(row['members'], evaluator.sites) for row in current['observations']}
                pending = []
                probability = {site:np.asarray(current['probability'][site]) for site in evaluator.sites}
                current['population'] += 1
                while len(pending) < min(search['population'], remaining):
                    group = sample_group(probability, evaluator, rng)
                    key = group_key(group, evaluator.sites)
                    duplicate = key in keys
                    current['proposals'].append(dict(number=len(current['proposals']), population=current['population'],
                        members=group, duplicate=duplicate))
                    current['duplicate_proposals'] += int(duplicate)
                    if not duplicate:
                        pending.append(group)
                        keys.add(key)
                current.update(pending=pending, pending_kind='population', rng_state=rng.bit_generator.state)
                save_state(state_path, state)
            batch = current['pending'][:config.get('group_batch_size', 4)]
            hidden, losses, conditional_losses = evaluator.evaluate(batch)
            for i, group in enumerate(batch):
                number = len(current['observations'])+1
                path = bank_dir/f'{method}_{number:04d}.npz'
                assert not path.exists()
                np.savez_compressed(path, pooled512=hidden[i], document_sha256=evaluator.documents,
                    members=np.array(json.dumps(group, sort_keys=True)), loss=np.array(losses[i]),
                    condition_losses=conditional_losses[i], status=np.array('PASS'),
                    method=np.array(method), query_number=np.array(number), protocol_sha256=np.array(protocol_hash))
                observation = dict(query_number=number, population=current['population'],
                    members=group, loss=float(losses[i]), condition_losses=conditional_losses[i].tolist(), path=str(path.resolve()))
                current['observations'].append(observation)
                if number in (32, 64, 128, budget):
                    current['snapshots'][str(number)] = best_observation(current['observations'], evaluator.sites)
            current['pending'] = current['pending'][len(batch):]
            if not current['pending'] and current['pending_kind'] == 'population':
                elite = sorted(current['observations'], key=lambda row: (row['loss'], group_key(row['members'], evaluator.sites)))[:search['elite']]
                if method == 'adaptive':
                    updated = {}
                    for site in evaluator.sites:
                        frequency = np.array([sum(int(member) in item['members'][site] for item in elite)
                            for member in evaluator.candidates[site]], dtype=np.float64)/len(elite)/evaluator.quotas[site]
                        updated[site] = (.5*np.array(current['probability'][site])+
                            .5*(.9*frequency+.1*evaluator.initial_probability[site])).tolist()
                        assert np.isclose(sum(updated[site]), 1.) and min(updated[site]) > 0
                    current['probability'] = updated
                current.setdefault('updates', []).append(dict(population=current['population'],
                    query_count=len(current['observations']), elite_queries=[item['query_number'] for item in elite],
                    probability=current['probability']))
            save_state(state_path, state)
            work.progress('GROUP_SEARCH', method=method, unique_groups=len(current['observations']),
                total_groups=budget, population=current['population'], duplicates=current['duplicate_proposals'],
                best_loss=best_observation(current['observations'], evaluator.sites)['loss'])
        assert len(current['observations']) == budget
        assert len({group_key(row['members'], evaluator.sites) for row in current['observations']}) == budget
        best = best_observation(current['observations'], evaluator.sites)
        selections[method] = best['members']
        summaries[method] = dict(best=best, snapshots=current['snapshots'],
            distinct_groups=budget, target_sequence_forwards=16*budget, duplicate_proposals=current['duplicate_proposals'])
        for observation in current['observations']:
            work.record(kind='group_calibration', task='whole_W', row_id=observation['query_number'],
                component=str(observation['query_number']), method=method, operation='both_backgrounds',
                seed=config['target_seed'], target_seed=config['target_seed'], split='calibration',
                loss=observation['loss'], population=observation['population'])
    write(work.run/'selected_members.json', selections)
    write(work.run/'prefix_selected_members.json', {f'{method}_n{count}':value['members']
        for method, data in summaries.items() for count,value in data['snapshots'].items()})
    write(work.run/'SEARCH_RESULTS.json', summaries)
    work.checks.update(same_candidate_pool=True, fixed_cardinality=True, binary_whole_group_execution=True,
        complete_distinct_query_budgets=True, calibration_only=True, shared_source_reference=True,
        finite_observed_losses=all(np.isfinite(row['loss']) for data in state['methods'].values() for row in data['observations']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    work = MultisiteWork(config, args.config, ['scripts/run_shift_group_feedback.py', 'scripts/run_shift_member_responses.py',
        'scripts/run_shift_explanation.py', 'scripts/train_shift_dictionaries.py', 'scripts/run_causalgym_multisite.py',
        'scripts/run_r011s1_raw_hook_asset.py', 'src/ccad/artifacts.py'])
    evaluator, error = None, None
    try:
        evaluator = GroupEvaluator(work, config)
        run_search(work, evaluator, config)
    except Exception:
        error = traceback.format_exc()
    finally:
        if evaluator is not None:
            evaluator.close()
    return work.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
