"""Apply a retained fixed native correspondence with NumPy only.

Inputs are codes of the actual recipient and donor states from the declared
target SAE. The artifact contains the complete signed reader and two fixed
nonnegative write supports. This module does not encode text or infer a new
operation. No source codes, source outcomes or target labels are required.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class CompiledAxis:
    reader: np.ndarray
    members: np.ndarray
    weights: np.ndarray
    directions: np.ndarray

    @classmethod
    def load(cls,path: str|Path):
        with np.load(path,allow_pickle=False) as z:
            result=cls(z['target_reader'].copy(),z['positive_negative_members'].copy(),
                       z['positive_negative_code_weights'].copy(),z['realized_unit_directions'].copy())
        r,j,w,v=result.reader,result.members,result.weights,result.directions
        if r.ndim!=1 or j.ndim!=2 or j.shape[0]!=2 or w.shape!=j.shape or v.ndim!=2 or v.shape[0]!=2:
            raise ValueError('Invalid compiled-axis shapes')
        if not np.issubdtype(j.dtype,np.integer) or j.min()<0 or j.max()>=len(r):
            raise ValueError('Write members outside the target dictionary')
        if any(len(np.unique(row))!=len(row) for row in j):raise ValueError('Repeated member in a fixed support')
        if not all(np.isfinite(a).all() for a in [r,w,v]) or (w<0).any():raise ValueError('Nonfinite or negative compiled weights')
        return result

    def apply(self,base_codes,donor_codes,*,decoder=None):
        """Return actual sparse code increments and the associated hidden edit.

        Shapes: codes [rows,members], decoder [members,hidden] if supplied.
        Without decoder, hidden deltas use the saved precomputed directions.
        With decoder, deltas are recomputed from the returned native increments;
        float32 CPU/GPU accumulation may differ from the saved cached vectors.
        """
        base=np.asarray(base_codes);donor=np.asarray(donor_codes)
        if base.ndim!=2 or donor.shape!=base.shape or base.shape[1]!=len(self.reader):raise ValueError('Codes must have matching [rows,target_members] shape')
        if not np.isfinite(base).all() or not np.isfinite(donor).all() or (base<0).any() or (donor<0).any():raise ValueError('Actual SAE states must be finite and nonnegative')
        alpha=(donor-base)@self.reader;sign=(alpha<0).astype(np.int64)
        members=self.members[sign];increment=np.abs(alpha)[:,None]*self.weights[sign]
        cached=np.abs(alpha)[:,None]*self.directions[sign]
        if decoder is None:delta=cached
        else:
            dec=np.asarray(decoder)
            if dec.shape!=(len(self.reader),self.directions.shape[1]) or not np.isfinite(dec).all():raise ValueError('Decoder must use target-member rows')
            delta=np.einsum('nb,nbd->nd',increment,dec[members],optimize=True)
        final_selected=np.take_along_axis(base,members,axis=1)+increment
        return dict(coordinate=alpha,members=members,code_increment=increment,
                    final_selected_codes=final_selected,hidden_delta=delta,
                    cached_hidden_delta=cached,changed_members=np.count_nonzero(increment,axis=1))
