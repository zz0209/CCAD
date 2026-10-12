import argparse
import json
import time

import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

DESCRIPTION = ('Download Gemma-3 and the Gemma Scope 2 PLT and CLT parameters referenced by circuit-tracer configs into '
               'the Hugging Face cache, printing bytes per file as progress.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--model', default='google/gemma-3-1b-pt')
    parser.add_argument('--configs', nargs='+', default=['mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine',
                                                         'mwhanna/gemma-scope-2-1b-pt/clt/width_262k_l0_medium_affine'])
    args = parser.parse_args()
    api = HfApi()
    info = api.model_info(args.model)
    print(json.dumps(dict(stage='access', model=args.model, sha=info.sha, gated=str(info.gated))), flush=True)
    snapshot_download(args.model, allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'])
    print(json.dumps(dict(stage='model_done')), flush=True)
    for ref in args.configs:
        owner, name, *sub = ref.split('/')
        folder = '/'.join(sub)
        # PLT folders keep the parameter list in config_orig.yaml (config.yaml points circuit-tracer at the folder);
        # CLT folders have only config.yaml with the list.
        files = {s.rfilename for s in api.model_info(f'{owner}/{name}').siblings}
        cfg_name = 'config_orig.yaml' if f'{folder}/config_orig.yaml' in files else 'config.yaml'
        cfg = yaml.safe_load(open(hf_hub_download(f'{owner}/{name}', f'{folder}/{cfg_name}')))
        paths = cfg['transcoders']
        t0 = time.time()
        total = 0
        for k, p in enumerate(paths):
            repo_path = p[len('hf://'):]
            o, n, *rest = repo_path.split('/')
            local = hf_hub_download(f'{o}/{n}', '/'.join(rest))
            total += __import__('os').path.getsize(local)
            print(json.dumps(dict(stage='file', ref=ref, k=k + 1, of=len(paths), gib=round(total / 2 ** 30, 2),
                                  seconds=round(time.time() - t0))), flush=True)
    print(json.dumps(dict(stage='done')), flush=True)


if __name__ == '__main__':
    main()
