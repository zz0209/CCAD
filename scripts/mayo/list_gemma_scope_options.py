from collections import defaultdict
import json
import sys

from huggingface_hub import HfApi

DESCRIPTION = 'List the widths and average L0 options that Gemma Scope releases at the 23 sites of the Gemma SHIFT explanation.'


def main():
    features = json.load(open(sys.argv[1]))['features']
    sites = sorted(features, key=lambda s: (s.split('_')[0], int(s.split('_')[1])))
    api = HfApi()
    out = {}
    for repo, kind in [('google/gemma-scope-2b-pt-res', 'resid'), ('google/gemma-scope-2b-pt-att', 'attn')]:
        info = api.model_info(repo)
        files = [f for f in api.list_repo_files(repo, revision=info.sha) if f.endswith('params.npz')]
        options = defaultdict(lambda: defaultdict(list))
        for f in files:
            parts = f.split('/')
            # Released dictionaries sit at layer_<n>/width_<w>/average_l0_<k>/params.npz.
            if len(parts) != 4 or not parts[0].startswith('layer_') or not parts[2].startswith('average_l0_'):
                continue
            options[int(parts[0].split('_')[1])][parts[1]].append(int(parts[2].split('_')[-1]))
        for site in sites:
            if site.startswith(kind):
                layer = int(site.split('_')[1])
                out[site] = {w: sorted(v) for w, v in sorted(options[layer].items())}
        out[f'{repo}@revision'] = info.sha
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
