from pathlib import Path
import argparse
import json
import sys
import traceback
import numpy as np
import torch
from ccad.program_counterparts import shared_support_order
from run_causalgym_multisite import MultisiteWork, write


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    config = json.loads((ROOT/settings['base_config']).read_text()) | settings
    work = MultisiteWork(config,args.config,['scripts/export_source_counterparts.py',
        'src/ccad/program_counterparts.py','scripts/run_causalgym_multisite.py',
        'scripts/run_r011s1_raw_hook_asset.py','src/ccad/artifacts.py'])
    error = None
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        work.torch = torch
        work.device = torch.device(config['device'])
        torch.cuda.set_device(work.device)
        torch.cuda.reset_peak_memory_stats(work.device)
        work.environment = dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__,device=config['device'])
        source = Path(config['source_action_bank'])
        arrays = dict(np.load(work.checked(source/'source_action_bank.npz')))
        metadata = json.loads(work.checked(source/'source_action_bank.json').read_text())
        all_supports = {}
        for seed in config['export_target_seeds']:
            parts = {part:{} for part in metadata['parts']}
            for site in metadata['sites']:
                path = Path(config['target_directory'])/f'{site}_seed{seed}.pt'
                state = torch.load(work.checked(path),map_location=work.device,weights_only=True)
                decoder = state['decoder.weight'].T
                for part in metadata['parts']:
                    covariance = torch.tensor(arrays[site+'__'+part],device=work.device)
                    indices = shared_support_order(decoder,covariance,config['counterpart_size']) if float(covariance.trace())>1e-20 else []
                    parts[part][site] = indices
                    work.record(kind='counterpart',task=site,component=part,row_id=f'{seed}_{site}_{part}',members=len(indices),seed=seed)
                work.progress('TARGET_COUNTERPART',seed=seed,site=site)
            scheme=config.get('counterpart_scheme','action_span')
            bank = dict(supports={f'{scheme}_{config["counterpart_size"]}':parts},target_seed=seed,
                source_identity=metadata['source_identity'],contexts=metadata['rows'],selected_requests=metadata['parts'],
                source_effect_kind=metadata.get('source_effect_kind','local_action'),
                selection='Shared squared-correlation pursuit on the fixed source-effect bank.',target_responses_used=False)
            write(work.run/f'counterparts_t{seed}.json',bank)
            all_supports[seed] = parts
        work.checks.update(source_only_bank=not metadata['target_dictionary_used'], target_responses_unused=True,
                           complete=len(all_supports)==len(config['export_target_seeds']))
    except BaseException:
        error = traceback.format_exc()
        raise
    finally:
        code = work.finish(error)
        if error is None and code:
            raise RuntimeError('Counterpart export validation failed')


if __name__ == '__main__':
    main()
