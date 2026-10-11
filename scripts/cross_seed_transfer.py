import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import numpy as np
import torch
import transformers
from scipy.optimize import linear_sum_assignment

from run_causalgym_multisite import MultisiteWork, write

DESCRIPTION = ('Carry a grouped SAE explanation from a source model and dictionary into a target model and dictionary '
               'with a fixed sparse membership, trained either on natural text with random feature subsets or on task '
               'part interventions, and compare its part effects with the source edit and one-to-one matching.')
SOURCES = ['scripts/cross_seed_transfer.py', 'scripts/run_causalgym_multisite.py', 'scripts/run_r011s1_raw_hook_asset.py',
           'src/ccad/artifacts.py']


class Side:
    # One language model with an SAE at one hook; an optional edit is added to the hook output.
    def __init__(self, w, spec, dictionary_class, device, model=None):
        if model is None:
            for name in spec.get('model_files', ['config.json', 'model.safetensors']):
                w.checked(Path(spec['model_dir'])/name, spec['model_label'])
            model = transformers.AutoModelForCausalLM.from_pretrained(
                spec['model_dir'], local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval().to(device)
            model.requires_grad_(False)
            model.config.use_cache = False
        self.model = model
        state = torch.load(w.checked(spec['sae_checkpoint'], spec['sae_label']), map_location=device, weights_only=True)
        self.sae = dictionary_class(self.model.config.hidden_size, state['encoder.weight'].shape[0], int(state['k'])).to(device)
        self.sae.load_state_dict(state)
        self.sae.eval().requires_grad_(False)
        self.edit, self.positions, self.codes, self.states = None, None, None, None
        self.model.get_submodule(spec['hook_module_path']).register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.codes is not None:
            self.codes.append(self.sae.encode(h.detach()))
        if self.states is not None:
            self.states.append(h.detach())
        if self.edit is None:
            return output
        delta = self.edit(h)
        if self.positions is not None:
            keep = torch.zeros(h.shape[:2], device=h.device)
            keep[torch.arange(len(h), device=h.device), self.positions] = 1.
            delta = delta*keep[..., None]
        h = h+delta
        return (h, *output[1:]) if isinstance(output, tuple) else h

    def logprobs(self, ids, mask=None):
        return self.model(ids, attention_mask=mask).logits.log_softmax(-1)


def margins(side, rows, eos):
    # Log-probability of the good sentence minus the bad one, with any edit at the last shared prefix token.
    length = max(len(r[key]) for r in rows for key in ['good', 'bad'])
    ids = torch.full((2*len(rows), length), eos, dtype=torch.long, device=side.model.device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        for j, key in enumerate(['good', 'bad']):
            ids[2*i+j, :len(row[key])] = torch.tensor(row[key], device=ids.device)
            mask[2*i+j, :len(row[key])] = 1
    side.positions = torch.tensor([r['position'] for r in rows], device=ids.device).repeat_interleave(2)
    lp = side.logprobs(ids, mask)
    side.positions = None
    token = lp[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
    total = (token*mask[:, 1:]).sum(1).reshape(-1, 2)
    return total[:, 0]-total[:, 1]


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    if args.smoke:
        c.update(run_id=c['run_id']+'_SMOKE', steps=4, correlation_sequences=8, natural_eval_sequences=4,
                 eval_pairs_per_task=4, fit_pairs_per_task=4)
    w = MultisiteWork(c, args.config, SOURCES)
    error = None
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        device = torch.device(c['device'])
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
        w.torch, w.device = torch, device
        sys.path.extend([c['dictionary_source_dir'], c['dictionary_overlay_dir']])
        from dictionary_learning.trainers.top_k import AutoEncoderTopK
        w.checked(Path(c['dictionary_source_dir'])/'dictionary_learning/trainers/top_k.py', 'Pinned TopK implementation', 'MIT')
        w.environment = dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__,
                             transformers=transformers.__version__)
        source = Side(w, c['source'], AutoEncoderTopK, device)
        # Within one language model the target SAE gets its own hook on the same module.
        target = Side(w, c['target'], AutoEncoderTopK, device, source.model if c['target'].get('same_model') else None)
        gate = torch.tensor(np.load(w.checked(c['source']['gate'], 'Source explanation groups'))['gate'], device=device)
        members = gate.sum(1).nonzero().flatten()
        parts = gate[members].argmax(1)
        m = len(members)
        dec_s = source.sae.decoder.weight[:, members].T
        dtype = c.get('token_dtype', '<u2')
        tokens = np.memmap(w.checked(c['natural_tokens'], 'Natural text tokens'), dtype=dtype, mode='r').reshape(-1, c['context'])
        valid = np.memmap(w.checked(c['natural_eval_tokens'], 'Natural validation tokens'), dtype=dtype, mode='r').reshape(-1, c['context'])
        # One-to-one families may rescale their single match up to this gain; memberships keep loads within one.
        gain_max = c.get('matched_gain_max', 1.)

        def batch(array, rows):
            return torch.tensor(np.asarray(array[rows], dtype=np.int64), device=device)

        decoy = any(f.startswith('decoy') for f in c['families'])
        tracked = members
        if decoy:
            # Decoy explanation: for each member, one source feature outside the explanation from the same source
            # activation-frequency bin; its candidates come from the same correlation rule as the members' own.
            fire, count = None, 0
            for off in range(0, c['decoy_frequency_sequences'], c['natural_batch']):
                ids = batch(tokens, slice(off, off+c['natural_batch']))
                source.codes = []
                with torch.no_grad():
                    source.model(ids)
                z = source.codes[0].reshape(-1, source.codes[0].shape[-1])
                source.codes = None
                fire = (z > 0).sum(0).double() if fire is None else fire+(z > 0).sum(0)
                count += z.shape[0]
            rate_s = (fire/count).cpu().numpy()
            bins_s = np.empty(len(rate_s), dtype=int)
            bins_s[np.argsort(rate_s, kind='stable')] = np.arange(len(rate_s))*c['capacity_bins']//len(rate_s)
            outside = (rate_s > 0) & ~np.isin(np.arange(len(rate_s)), members.cpu().numpy())
            drng = np.random.default_rng(c['training_seed']+13)
            chosen = []
            for i in members.tolist():
                chosen.append(int(drng.choice([u for u in np.flatnonzero((bins_s == bins_s[i]) & outside) if u not in chosen])))
            decoys = torch.tensor(chosen, device=device)
            tracked = torch.cat([members, decoys])
        mt = len(tracked)
        # Natural-text activation statistics between the tracked source features and every target feature.
        sums = dict(n=0., a=torch.zeros(mt, device=device, dtype=torch.float64), b=None, aa=torch.zeros(mt, device=device, dtype=torch.float64),
                    bb=None, ab=None, nz=None)
        # Dense reference: an affine least-squares map from source to target hook states on the same natural text.
        dense = c.get('dense_stitch', False) and target.model is not source.model
        dim = source.model.config.hidden_size
        if dense:
            xtx = torch.zeros(dim+1, dim+1, device=device, dtype=torch.float64)
            xty = torch.zeros(dim+1, target.model.config.hidden_size, device=device, dtype=torch.float64)
        for off in range(0, c['correlation_sequences'], c['natural_batch']):
            ids = batch(tokens, slice(off, off+c['natural_batch']))
            source.codes, target.codes = [], []
            if dense:
                source.states, target.states = [], []
            with torch.no_grad():
                source.model(ids)
                if target.model is not source.model:
                    target.model(ids)
            za = source.codes[0][..., tracked].reshape(-1, mt).double()
            zb = target.codes[0].reshape(za.shape[0], -1).double()
            source.codes = target.codes = None
            if dense:
                xa = source.states[0].reshape(-1, dim).double()
                xa = torch.cat([xa, torch.ones(len(xa), 1, device=device, dtype=torch.float64)], 1)
                xtx += xa.T@xa
                xty += xa.T@target.states[0].reshape(len(xa), -1).double()
                source.states = target.states = None
            if sums['b'] is None:
                n_t = zb.shape[1]
                sums.update(b=torch.zeros(n_t, device=device, dtype=torch.float64), bb=torch.zeros(n_t, device=device, dtype=torch.float64),
                            ab=torch.zeros(mt, n_t, device=device, dtype=torch.float64), nz=torch.zeros(n_t, device=device, dtype=torch.float64))
            sums['n'] += za.shape[0]
            sums['nz'] += (zb > 0).sum(0)
            sums['a'] += za.sum(0)
            sums['b'] += zb.sum(0)
            sums['aa'] += (za*za).sum(0)
            sums['bb'] += (zb*zb).sum(0)
            sums['ab'] += za.T@zb
            w.progress('CORRELATION', items=min(off+c['natural_batch'], c['correlation_sequences']), total=c['correlation_sequences'])
        n = sums['n']
        cov = sums['ab']/n-torch.outer(sums['a']/n, sums['b']/n)
        sd_a = (sums['aa']/n-(sums['a']/n)**2).clamp_min(1e-12).sqrt()
        sd_b = (sums['bb']/n-(sums['b']/n)**2).clamp_min(1e-12).sqrt()
        corr_all = (cov/torch.outer(sd_a, sd_b)).float()
        corr = corr_all[:m]
        corr_cpu = corr.cpu().numpy()
        rows_h, cols_h = linear_sum_assignment(-corr_cpu)
        hungarian = torch.tensor(cols_h[np.argsort(rows_h)], device=device)
        nearest = corr.topk(c['candidates_per_source'], dim=1).indices
        cand = torch.unique(torch.cat([nearest.flatten(), hungarian]))
        position = {int(t): i for i, t in enumerate(cand.tolist())}
        allowed = torch.zeros(len(cand), m, device=device)
        for i in range(m):
            allowed[[position[int(t)] for t in nearest[i].tolist()], i] = 1.
            allowed[position[int(hungarian[i])], i] = 1.
        matched = torch.zeros_like(allowed)
        for i in range(m):
            matched[position[int(hungarian[i])], i] = 1.
        dec_t = target.sae.decoder.weight[:, cand].T
        if decoy:
            # Candidates and Hungarian start of the decoys, assigned to the members they stand in for.
            corr_d = corr_all[m:]
            rows_d, cols_d = linear_sum_assignment(-corr_d.cpu().numpy())
            hung_d = torch.tensor(cols_d[np.argsort(rows_d)], device=device)
            near_d = corr_d.topk(c['candidates_per_source'], dim=1).indices
            cand_dc = torch.unique(torch.cat([near_d.flatten(), hung_d]))
            pos_d = {int(t): j for j, t in enumerate(cand_dc.tolist())}
            allowed_d = torch.zeros(len(cand_dc), m, device=device)
            matched_d = torch.zeros_like(allowed_d)
            for i in range(m):
                allowed_d[[pos_d[int(t)] for t in near_d[i].tolist()], i] = 1.
                allowed_d[pos_d[int(hung_d[i])], i] = 1.
                matched_d[pos_d[int(hung_d[i])], i] = 1.
            dec_dc = target.sae.decoder.weight[:, cand_dc].T
            write(w.run/'DECOY.json', dict(
                bins=c['capacity_bins'], frequency_sequences=c['decoy_frequency_sequences'],
                member_rate=float(rate_s[members.cpu().numpy()].mean()), decoy_rate=float(rate_s[decoys.cpu().numpy()].mean()),
                member_hungarian_corr=float(corr[torch.arange(m), hungarian].mean()),
                decoy_hungarian_corr=float(corr_d[torch.arange(m), hung_d].mean()),
                decoy_candidates=len(cand_dc), real_candidates=len(cand),
                shared_candidates=int(np.isin(cand_dc.cpu().numpy(), cand.cpu().numpy()).sum())))
        if any(f.startswith('capacity') for f in c['families']):
            # Capacity control: each candidate of each source feature is replaced by a random target feature from the
            # same activation-frequency bin, outside every real candidate, so the map keeps its size and firing rates
            # but loses the features chosen for the explanation.
            rate = (sums['nz']/n).cpu().numpy()
            bins = np.empty(len(rate), dtype=int)
            bins[np.argsort(rate, kind='stable')] = np.arange(len(rate))*c['capacity_bins']//len(rate)
            free = ~np.isin(np.arange(len(rate)), cand.cpu().numpy())
            pools = [np.flatnonzero((bins == b) & free) for b in range(c['capacity_bins'])]
            crng = np.random.default_rng(c['training_seed']+11)
            lists, match_r = [], []
            for i in range(m):
                real = list(dict.fromkeys(nearest[i].tolist()+[int(hungarian[i])]))
                chosen = {}
                for t in real:
                    pool = [u for u in pools[bins[t]] if u not in chosen.values()]
                    chosen[t] = int(crng.choice(pool))
                lists.append(list(chosen.values()))
                match_r.append(chosen[int(hungarian[i])])
            cand_r = torch.tensor(sorted({u for r in lists for u in r}), device=device)
            pos_r = {int(t): j for j, t in enumerate(cand_r.tolist())}
            allowed_r = torch.zeros(len(cand_r), m, device=device)
            matched_r = torch.zeros_like(allowed_r)
            for i in range(m):
                allowed_r[[pos_r[u] for u in lists[i]], i] = 1.
                matched_r[pos_r[match_r[i]], i] = 1.
            dec_r = target.sae.decoder.weight[:, cand_r].T
            write(w.run/'CAPACITY.json', dict(bins=c['capacity_bins'], candidates=len(cand_r), real_candidates=len(cand),
                                              real_rate=float(rate[cand.cpu().numpy()].mean()),
                                              replacement_rate=float(rate[cand_r.cpu().numpy()].mean())))
        if dense:
            ridge = c['dense_ridge']*xtx.diagonal().mean()*torch.eye(dim+1, device=device, dtype=torch.float64)
            stitch = torch.linalg.solve(xtx+ridge, xty)[:dim].float()
        np.savez_compressed(w.run/'correlation.npz', members=members.cpu().numpy(), parts=parts.cpu().numpy(),
                            hungarian=hungarian.cpu().numpy(), hungarian_corr=corr_cpu[np.arange(m), hungarian.cpu().numpy()],
                            nearest=nearest.cpu().numpy(), nearest_corr=corr.gather(1, nearest).cpu().numpy())

        def source_edit(q):
            def edit(h):
                z = source.sae.encode(h)[..., members]
                qq = q[:, None, :] if q.dim() == 2 else q
                return -(z*qq)@dec_s
            return edit

        def membership_edit(weights, q, cap=1., cset=None):
            idx, dec = (cand, dec_t) if cset is None else cset

            def edit(h):
                z = target.sae.encode(h)[..., idx]
                load = q@weights.T if cap is None else (q@weights.T).clamp(max=cap)
                load = load[:, None, :] if q.dim() == 2 else load
                return -(z*load)@dec
            return edit

        panel = json.loads(w.checked(c['panel'], 'Fit panel').read_text())
        fit = [r for task in c['tasks'] for r in [r for r in panel['rows'] if r['task'] == task and r['split'] == 'fit'][:c['fit_pairs_per_task']]]
        evaluation = json.loads(w.checked(c['evaluation_panel'], 'Evaluation panel').read_text())
        rows = [r for task in c['tasks'] for r in [r for r in evaluation['rows'] if r['task'] == task and
                r['split'] == c['evaluation_split']][:c['eval_pairs_per_task']]]
        assert not {r['sentence_good'] for r in fit} & {r['sentence_good'] for r in rows}
        queries = c['queries']
        eos = c['eos_token_id']
        write(w.run/'panel.json', dict(fit=fit, rows=rows, queries=queries, query_order=list(queries)))

        @torch.no_grad()
        def evaluate_task(name, side, make_edit):
            out = np.empty((len(queries), len(rows)))
            for qi, vector in enumerate(queries.values()):
                q = torch.tensor(vector, device=device, dtype=torch.float32)[parts]
                for off in range(0, len(rows), c['eval_batch_pairs']):
                    rr = rows[off:off+c['eval_batch_pairs']]
                    side.edit = None if make_edit is None else make_edit(q)
                    out[qi, off:off+len(rr)] = margins(side, rr, eos).cpu().numpy()
                    side.edit = None
            record_task(name, out)
            return out

        def record_task(name, out):
            w.records([dict(kind='task', task=r['task'], row_id=r['row_id'], component=f"{r['task']}:{r['row_id']}",
                            method=name, operation=query, mode=query,
                            seed=c['seeds'][0], target_seed=c['seeds'][1], margin=float(out[qi, j]))
                       for qi, query in enumerate(queries) for j, r in enumerate(rows)])

        @torch.no_grad()
        def evaluate_dense_task():
            # The source edit computed in the source model and carried into the target hook by the dense map.
            out = np.empty((len(queries), len(rows)))
            for qi, vector in enumerate(queries.values()):
                q = torch.tensor(vector, device=device, dtype=torch.float32)[parts]
                for off in range(0, len(rows), c['eval_batch_pairs']):
                    rr = rows[off:off+c['eval_batch_pairs']]
                    source.states = []
                    margins(source, rr, eos)
                    mapped = (-(source.sae.encode(source.states[0])[..., members]*q)@dec_s)@stitch
                    source.states = None
                    target.edit = lambda h, d=mapped: d
                    out[qi, off:off+len(rr)] = margins(target, rr, eos).cpu().numpy()
                    target.edit = None
            record_task('dense_stitch', out)
            return out

        @torch.no_grad()
        def evaluate_dense_natural():
            gen = torch.Generator(device='cpu').manual_seed(c['natural_eval_seed'])
            num = den = 0.
            for off in range(0, c['natural_eval_sequences'], c['natural_batch']):
                ids = batch(valid, slice(off, off+c['natural_batch']))
                p = torch.rand(len(ids), 1, generator=gen)
                q = (torch.rand(len(ids), m, generator=gen) < p).float().to(device)
                source.states = []
                lp_s0 = source.logprobs(ids)
                h_a = source.states[0]
                source.states = None
                source.edit = source_edit(q)
                d_s = source.logprobs(ids)-lp_s0
                source.edit = None
                lp_t0 = target.logprobs(ids)
                weight = .5*(lp_s0.exp()+lp_t0.exp())
                mapped = (-(source.sae.encode(h_a)[..., members]*q[:, None, :])@dec_s)@stitch
                target.edit = lambda h, d=mapped: d
                d_t = target.logprobs(ids)-lp_t0
                target.edit = None
                num += float((weight*(d_t-d_s)**2).sum())
                den += float((weight*d_s**2).sum())
            return float(np.sqrt(num/den))

        @torch.no_grad()
        def evaluate_natural(name, weights, cap=1., cset=None):
            # Probability-weighted error of next-token log-probability changes on validation text under random subsets.
            gen = torch.Generator(device='cpu').manual_seed(c['natural_eval_seed'])
            num = den = 0.
            for off in range(0, c['natural_eval_sequences'], c['natural_batch']):
                ids = batch(valid, slice(off, off+c['natural_batch']))
                p = torch.rand(len(ids), 1, generator=gen)
                q = (torch.rand(len(ids), m, generator=gen) < p).float().to(device)
                err, eff = natural_error(ids, q, weights, cap, cset)
                num += float(err)
                den += float(eff)
            return float(np.sqrt(num/den))

        def natural_error(ids, q, weights, cap=1., cset=None):
            with torch.no_grad():
                source.edit = None
                lp_s0 = source.logprobs(ids)
                source.edit = source_edit(q)
                d_s = source.logprobs(ids)-lp_s0
                source.edit = None
                lp_t0 = lp_s0 if target.model is source.model else target.logprobs(ids)
                weight = .5*(lp_s0.exp()+lp_t0.exp())
            target.edit = membership_edit(weights, q, cap, cset)
            d_t = target.logprobs(ids)-lp_t0
            target.edit = None
            return (weight*(d_t-d_s)**2).sum(), (weight*d_s**2).sum()

        def task_dist_error(rr, q, weights, cap=1.):
            # Reader-complete task error: the edit sits at the last shared prefix token of both sentences of each pair,
            # and the full next-token distribution is compared at that position and every later one.
            length = max(len(r[key]) for r in rr for key in ['good', 'bad'])
            ids = torch.full((2*len(rr), length), eos, dtype=torch.long, device=device)
            pad = torch.zeros_like(ids)
            for i, row in enumerate(rr):
                for j, key in enumerate(['good', 'bad']):
                    ids[2*i+j, :len(row[key])] = torch.tensor(row[key], device=device)
                    pad[2*i+j, :len(row[key])] = 1
            positions = torch.tensor([r['position'] for r in rr], device=device).repeat_interleave(2)
            keep = ((torch.arange(length, device=device)[None, :] >= positions[:, None]) & pad.bool()).float()
            qq = q.repeat_interleave(2, 0)
            with torch.no_grad():
                source.positions = positions
                lp_s0 = source.logprobs(ids, pad)
                source.edit = source_edit(qq)
                d_s = source.logprobs(ids, pad)-lp_s0
                source.edit = source.positions = None
                lp_t0 = lp_s0 if target.model is source.model else target.logprobs(ids, pad)
                weight = .5*(lp_s0.exp()+lp_t0.exp())*keep[..., None]
            target.positions = positions
            target.edit = membership_edit(weights, qq, cap)
            d_t = target.logprobs(ids, pad)-lp_t0
            target.edit = target.positions = None
            return (weight*(d_t-d_s)**2).sum(), (weight*d_s**2).sum()

        values = dict(none_source=evaluate_task('none_source', source, None),
                      source=evaluate_task('source', source, source_edit))
        values['none_target'] = values['none_source'] if target.model is source.model else evaluate_task('none_target', target, None)
        natural = {}
        maps = dict(hungarian_activation=matched)
        values['hungarian_activation'] = evaluate_task('hungarian_activation', target, lambda q: membership_edit(matched, q))
        natural['hungarian_activation'] = evaluate_natural('hungarian_activation', matched)
        if c['target'].get('gate'):
            # The target dictionary's own explanation, selected by the same rule, deleted part by part: what
            # rediscovery in the target gives, without any correspondence.
            gate_t = torch.tensor(np.load(w.checked(c['target']['gate'], 'Target explanation groups'))['gate'], device=device)
            members_t = gate_t.sum(1).nonzero().flatten()
            parts_t = gate_t[members_t].argmax(1)
            dec_tm = target.sae.decoder.weight[:, members_t].T
            first = torch.stack([(parts == k).nonzero()[0, 0] for k in range(int(parts.max())+1)])

            def rediscovered_edit(q):
                q_t = q[first][parts_t]

                def edit(h):
                    return -(target.sae.encode(h)[..., members_t]*q_t)@dec_tm
                return edit
            values['rediscovered'] = evaluate_task('rediscovered', target, rediscovered_edit)
        if dense:
            values['dense_stitch'] = evaluate_dense_task()
            natural['dense_stitch'] = evaluate_dense_natural()
        np.savez_compressed(w.run/'responses.npz', **values)
        w.progress('BASELINES', natural=natural)
        endpoints = torch.cat([torch.eye(3, device=device), torch.ones(1, 3, device=device)])
        for family in c['families']:
            capacity, decoyed = family.startswith('capacity'), family.startswith('decoy')
            mask = matched if family.startswith('matched') else (allowed_r if capacity else allowed_d if decoyed else allowed)
            hi, cap = (gain_max, None) if family.startswith('matched') else (1., 1.)
            cset = (cand_r, dec_r) if capacity else (cand_dc, dec_dc) if decoyed else None
            param = torch.nn.Parameter((matched_r if capacity else matched_d if decoyed else matched).clone())
            optimizer = torch.optim.AdamW([param], lr=c['lr'], weight_decay=0.)
            rng = np.random.default_rng(c['training_seed'])
            gen = torch.Generator(device='cpu').manual_seed(c['training_seed'])
            natural_steps = task_steps = 0
            for step in range(c['steps']):
                weights = (param*mask).clamp(0., hi)
                # Mixed families alternate natural-text subsets and task parts, each with its own schedule.
                if family.endswith('natural') or (family.endswith('mixed') and step % 2 == 0):
                    start = c['correlation_sequences']+natural_steps*c['natural_batch']
                    natural_steps += 1
                    ids = batch(tokens, slice(start, start+c['natural_batch']))
                    p = torch.rand(len(ids), 1, generator=gen)
                    q = (torch.rand(len(ids), m, generator=gen) < p).float().to(device)
                    err, eff = natural_error(ids, q, weights, cap, cset)
                    loss = err/eff.clamp_min(1e-12)
                elif family.endswith('distparts'):
                    # Reader-complete part supervision: part and complete requests alternate with per-pair random
                    # feature subsets, all read by the full next-token distribution of the task sentences.
                    k = task_steps
                    task_steps += 1
                    rr = [fit[i] for i in rng.choice(len(fit), c['batch_pairs'], replace=False)]
                    if k % 2 == 0:
                        q = endpoints[(k//2) % len(endpoints)][parts].expand(len(rr), -1)
                    else:
                        p = torch.rand(len(rr), 1, generator=gen)
                        q = (torch.rand(len(rr), m, generator=gen) < p).float().to(device)
                    err, eff = task_dist_error(rr, q, weights, cap)
                    loss = err/eff.clamp_min(1e-12)
                else:
                    k = task_steps
                    task_steps += 1
                    if family.endswith('complete'):
                        request = endpoints[-1]
                    elif k % 2 == 0:
                        request = endpoints[(k//2) % len(endpoints)]
                    else:
                        request = torch.tensor(rng.random(3), device=device, dtype=torch.float32)
                    q = request[parts]
                    rr = [fit[i] for i in rng.choice(len(fit), c['batch_pairs'], replace=False)]
                    with torch.no_grad():
                        source.edit = source_edit(q)
                        effect_s = margins(source, rr, eos)
                        source.edit = None
                        base_s = margins(source, rr, eos)
                        base_t = base_s if target.model is source.model else margins(target, rr, eos)
                    target.edit = membership_edit(weights, q, cap, cset)
                    effect_t = margins(target, rr, eos)
                    target.edit = None
                    loss = ((effect_t-base_t)-(effect_s-base_s)).square().mean()/(effect_s-base_s).square().mean().clamp_min(1e-12)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    param.clamp_(0., hi)
                assert bool(torch.isfinite(loss))
                if (step+1) % c['progress_every'] == 0 or step+1 == c['steps']:
                    w.progress('TRAINING', method=family, step=step+1, total=c['steps'], loss=float(loss.detach()))
            weights = (param.detach()*mask).clamp(0., hi)
            maps[family] = weights
            values[family] = evaluate_task(family, target, lambda q, wt=weights, cp=cap, cs=cset: membership_edit(wt, q, cp, cs))
            natural[family] = evaluate_natural(family, weights, cap, cset)
            np.savez_compressed(w.run/'responses.npz', **values)
            w.progress('EVALUATED', method=family, natural=natural[family])
        extra = dict(capacity_candidates=cand_r.cpu().numpy(), capacity_allowed=allowed_r.cpu().numpy()) if any(
            f.startswith('capacity') for f in c['families']) else {}
        if decoy:
            extra.update(decoys=decoys.cpu().numpy(), decoy_candidates=cand_dc.cpu().numpy(), decoy_allowed=allowed_d.cpu().numpy())
        np.savez_compressed(w.run/'maps.npz', candidates=cand.cpu().numpy(), members=members.cpu().numpy(),
                            parts=parts.cpu().numpy(), allowed=allowed.cpu().numpy(), **extra,
                            **{k: v.cpu().numpy() for k, v in maps.items()})
        write(w.run/'NATURAL.json', natural)
        w.checks['all_families_evaluated'] = all(f in values for f in c['families'])
    except Exception:
        error = traceback.format_exc()
    return w.finish(error)


if __name__ == '__main__':
    raise SystemExit(main())
