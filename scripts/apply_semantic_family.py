"""Apply a saved source family to a target decoded difference, optionally write it.

NumPy only. The optional support is fixed by the caller: this command solves
its bounded decoder-space fit and preserves the original reconstruction
residual. It does not perform adaptive dictionary search or LM inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ccad.semantic_readout import apply_numpy
from ccad.native_operation import project_native


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--basis', type=Path, required=True, help='NPZ with bases [attribute,rank,hook]')
    ap.add_argument('--difference', type=Path, required=True, help='NPZ with decoded_difference [batch,hook]')
    ap.add_argument('--controls', required=True, help='Comma-separated strengths in stored attribute order')
    ap.add_argument('--native-support', type=Path, help='NPZ with decoder [members,hook], base_codes [batch,members], members')
    ap.add_argument('--max-steps', type=int, default=2000)
    ap.add_argument('--tolerance', type=float, default=1e-7)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep prior outputs; choose a new output path')
    if args.max_steps < 1 or not np.isfinite(args.tolerance) or args.tolerance <= 0:
        raise ValueError('Positive iteration limit and tolerance required')
    bases = np.load(args.basis, allow_pickle=False)['bases']
    difference = np.load(args.difference, allow_pickle=False)['decoded_difference']
    controls = np.asarray([float(v) for v in args.controls.split(',')])
    q = apply_numpy(difference, bases, controls)
    output = dict(readout_delta=q, controls=controls)
    info = dict(readout_shape=list(q.shape), attribute_order='Stored basis order; R18 is Country,Continent,Language',
        scope='Apply readout_delta or native_delta at the recorded recipient hook position. Preserve original residual, untouched codes and all other positions. No LM outputs are produced.')
    inputs = [args.basis, args.difference]
    if args.native_support:
        support = np.load(args.native_support, allow_pickle=False)
        d, z, members = support['decoder'], support['base_codes'], support['members']
        if d.ndim != 2 or z.ndim != 2 or members.shape != (len(d),) or len(np.unique(members)) != len(members):
            raise ValueError('Expected one fixed support with unique member identities')
        if not np.isfinite(d).all() or not np.isfinite(z).all():
            raise ValueError('Nonfinite native support')
        u, diag = project_native(q, z, d, max_steps=args.max_steps, tolerance=args.tolerance)
        output.update(native_delta=u@d, code_increment=u, final_selected_codes=z+u,
                      members=members, member_contributions=u[:, :, None]*d[None, :, :])
        info['native_solver'] = diag
        info['native_scope'] = 'Fixed supplied support; the bounded optimum may have nonzero error. These codes need not be a TopK encoder output.'
        inputs.append(args.native_support)
    info['inputs'] = [dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dict.fromkeys(inputs)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output, provenance_json=json.dumps(info))
    print(json.dumps(dict(output=str(args.output.resolve()), **info)))


if __name__ == '__main__':
    main()
