from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np


def main():
    root = Path('D:/CCAD_Storage/runs/reuse_generalization_20260921_round04')
    smoke = root/'RG04_NUMBER_COUNTERPART_SMOKE_T2_20260921'
    replay = root/'RG04_NUMBER_COUNTERPART_REPLAY_SMOKE_T2_20260921'
    selected = root/'RG04_NUMBER_COUNTERPART_SELECTION_T2_20260921'
    a, b = dict(np.load(smoke/'responses.npz')), dict(np.load(replay/'responses.npz'))
    equality = {key:bool(np.array_equal(value,b[key])) for key,value in a.items()}
    if not all(equality.values()):
        raise ValueError(equality)
    stats = dict(np.load(selected/'counterpart_statistics.npz'))
    bank = json.loads((selected/'counterparts.json').read_text())
    checks = {}
    for key, value in stats.items():
        if key.endswith('__participation'):
            if value.min() < 0:
                raise ValueError(key)
            checks[key] = dict(sum=float(value.sum()), positive_members=int((value>0).sum()))
        else:
            if np.max(np.abs(value-value.T)) > 1e-12:
                raise ValueError(key)
            eigenvalues = np.linalg.eigvalsh(value)
            if eigenvalues[0] < -1e-8:
                raise ValueError(key)
            energy = eigenvalues.sum()
            rank90 = int(np.searchsorted(np.cumsum(eigenvalues[::-1]), .9*energy)+1) if energy > 1e-12 else 0
            checks[key] = dict(trace=float(energy), minimum_eigenvalue=float(eigenvalues[0]), rank90=rank90)
    for scheme, parts in bank['supports'].items():
        size = int(scheme.rsplit('_',1)[1])
        for part, sites in parts.items():
            for site, indices in sites.items():
                width = len(stats[site+'__'+part+'__participation'])
                if len(indices)>size or len(set(indices))!=len(indices) or any(i<0 or i>=width for i in indices):
                    raise ValueError((scheme,part,site))
    out = Path('artifacts/reuse_generalization_20260921_round04/NUMERICAL_CHECK.json')
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(dict(written_at_utc=datetime.now(timezone.utc).isoformat(),
        exact_smoke_replay=equality, selection_statistics=checks, support_index_checks=True),indent=2)+'\n')
    print(json.dumps(dict(exact_smoke_replay=equality, support_index_checks=True, output=str(out))))


if __name__ == '__main__':
    main()
