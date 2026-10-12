import argparse
import json
from pathlib import Path

import numpy as np
import torch

DESCRIPTION = ('Similarity of source supernode features to their best-correlated target features in saved P1 maps: '
               'distribution of the top activation correlation per source feature.')


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('maps', type=Path, nargs='+')
    args = parser.parse_args()
    for path in args.maps:
        m = torch.load(path, map_location='cpu', weights_only=False)
        top = np.array([v[1][0] for v in m['cand'].values()])
        print(json.dumps(dict(run=path.parent.name, features=len(top), median=round(float(np.median(top)), 3),
                              q25=round(float(np.quantile(top, .25)), 3), q75=round(float(np.quantile(top, .75)), 3),
                              above_09=round(float(np.mean(top > .9)), 3))))


if __name__ == '__main__':
    main()
