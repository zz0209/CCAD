from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json

from huggingface_hub import HfApi, snapshot_download
import torch
import transformers

DESCRIPTION = ('Download language models trained with the same recipe and different seeds, pin their commits, and '
               'save each as a standard local copy with model.safetensors and tokenizer.json for the CCAD runners.')
MODELS = {
    'openai-community/gpt2-medium': dict(revision='6dcaa7a952f72f9298047fd5137cd6e4f05f41da', folder='gpt2-medium-openai'),
    'stanford-crfm/arwen-gpt2-medium-x21': dict(revision=None, folder='gpt2-medium-x21'),
    'stanford-crfm/beren-gpt2-medium-x49': dict(revision=None, folder='gpt2-medium-x49'),
    'EleutherAI/pythia-70m-seed1': dict(revision=None, folder='pythia-70m-seed1'),
    'EleutherAI/pythia-70m-seed2': dict(revision=None, folder='pythia-70m-seed2'),
}
PATTERNS = ['config.json', 'pytorch_model.bin', 'model.safetensors', 'vocab.json', 'merges.txt', 'tokenizer.json',
            'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--root', type=Path, default=Path('/mnt/nvme3-data/harvardhds/ccad-m364569/assets/seed_models'))
    parser.add_argument('--only', nargs='*', default=list(MODELS))
    args = parser.parse_args()
    api = HfApi()
    manifest_path = args.root/'SEED_MODELS_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for repo in args.only:
        spec = MODELS[repo]
        info = api.model_info(repo, revision=spec['revision'])
        raw = args.root/'raw'/spec['folder']
        snapshot_download(repo, revision=info.sha, local_dir=raw, allow_patterns=PATTERNS)
        # One standard copy per model: safetensors weights and a fast tokenizer file.
        model = transformers.AutoModelForCausalLM.from_pretrained(raw, dtype=torch.float32)
        tokenizer = transformers.AutoTokenizer.from_pretrained(raw)
        local = args.root/spec['folder']
        model.save_pretrained(local, safe_serialization=True)
        tokenizer.save_pretrained(local)
        manifest[repo] = dict(commit=info.sha, license=(info.card_data or {}).get('license') if info.card_data else None,
                              raw_files={p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(raw.iterdir()) if p.is_file()},
                              local_dir=str(local),
                              local_files={p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(local.iterdir()) if p.is_file()},
                              written_utc=datetime.now(timezone.utc).isoformat())
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(json.dumps(dict(repo=repo, commit=info.sha, local=str(local))), flush=True)


if __name__ == '__main__':
    main()
