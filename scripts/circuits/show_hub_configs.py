import argparse

from huggingface_hub import HfApi, hf_hub_download

DESCRIPTION = 'Print the circuit-tracer config files of transcoder repositories and the sizes of the files they reference.'


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('refs', nargs='+', help='repo_id/subfolder, e.g. mwhanna/gemma-scope-2-270m-pt/clt/width_262k_l0_medium_affine')
    args = parser.parse_args()
    api = HfApi()
    for ref in args.refs:
        owner, name, *sub = ref.split('/')
        repo, folder = f'{owner}/{name}', '/'.join(sub)
        files = [s.rfilename for s in api.model_info(repo).siblings if s.rfilename.startswith(folder + '/')]
        print('==', ref, files)
        for f in files:
            if f.endswith(('.yaml', '.yml', '.json')):
                print(open(hf_hub_download(repo, f)).read()[:1500])


if __name__ == '__main__':
    main()
