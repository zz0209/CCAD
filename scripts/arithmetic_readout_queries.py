"""Use fitted source-amplitude predictors for the frozen member requests."""
import json
import hashlib


def prepare_readouts(w, cfg):
    import numpy as np
    import torch
    spec = cfg['query_readouts']; parent = w.run.parent.parent/spec['fit_run']
    assert json.loads(w.checked(parent/'status.json').read_text())['status'] == 'PASS'
    fitcfg = json.loads(w.checked(parent/'config.resolved.json').read_text())
    assert all(fitcfg[k] == cfg[k] for k in ['model_revision', 'training_run', 'checkpoint_step'])
    assert fitcfg['readout_fit']['relation_run'] == cfg['member_queries']['relation_run']
    query = json.loads((w.run/'MEMBER_QUERY_FREEZE.json').read_text())
    result, metadata = {}, []
    for record in query['banks'][0]['records']:
        s, t = record['source_seed'], record['target_seed']
        path = w.checked(parent/f'readout_s{s}_t{t}.npz')
        if spec.get('sha256'):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256'][path.name]
        with np.load(path) as z:
            shared = {}
            for op in ['unit', 'tens']:
                shared[op] = dict(decoder=torch.tensor(z[f'{op}_decoder'], device=w.device),
                    indices=torch.tensor(z[f'{op}_target_indices'], device=w.device),
                    raw=torch.tensor(z[f'{op}_raw'], device=w.device),
                    activation=torch.tensor(z[f'{op}_activation'], device=w.device),
                    source_indices=z[f'{op}_source_indices'])
            for bank, bankmeta in enumerate(query['banks']):
                row = next(r for r in bankmeta['records'] if r['source_seed']==s)
                for part in [0, 1]:
                    for kind in ['raw', 'activation']:
                        operations = {}
                        for item in row['queries']:
                            op = item['operation']; data = shared[op]
                            assert item['source_indices'] == data['source_indices'].tolist()
                            q = np.isin(data['source_indices'], item['part0_indices']).astype(np.float32)
                            if part:
                                q = 1-q
                            operations[op] = dict(kind=kind, coef=data[kind], decoder=data['decoder'],
                                indices=data['indices'], q=torch.tensor(q, device=w.device))
                        result[t, f'{kind}_readout_part{part}_bank{bank}', cfg['members'][0]] = operations
            for kind in ['raw', 'activation']:
                result[t, f'{kind}_readout_full', cfg['members'][0]] = {
                    op: dict(kind=kind, coef=d[kind], decoder=d['decoder'], indices=d['indices'],
                             q=torch.ones(len(d['source_indices']), device=w.device)) for op, d in shared.items()}
        metadata.append(dict(source_seed=s, target_seed=t, path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             raw_input='full hidden difference', activation_input='64 selected target code differences',
                             output='64 source decoder columns per operation, queried with the same source membership',
                             target_output_fits=0))
    return result, metadata
