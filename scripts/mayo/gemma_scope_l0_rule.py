import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

DESCRIPTION = ('List Gemma Scope SAEs chosen by the SFC rule (width 16k, average L0 closest to 100) and by the '
               'canonical registry at every site of the published Gemma SHIFT explanation.')


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text())
    api = HfApi()
    out = {}
    for repo, kind in [('google/gemma-scope-2b-pt-res', 'resid'), ('google/gemma-scope-2b-pt-att', 'attn')]:
        files = api.list_repo_files(repo)
        for site in manifest['features']:
            if not site.startswith(kind):
                continue
            layer = int(site.split('_')[1])
            for width in ['16k', '65k']:
                prefix = f'layer_{layer}/width_{width}/'
                options = sorted({f.split('/')[2] for f in files if f.startswith(prefix) and f.endswith('params.npz')})
                l0s = [(o, int(o.split('_')[-1])) for o in options]
                rule = min(l0s, key=lambda x: abs(x[1] - 100))[0]
                out[f'{site}/{width}'] = dict(repo=repo, options=options, sfc_rule=prefix + rule)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
