"""NumPy-only consumer of a frozen compact, rectified source operation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RectifiedOperation:
    target_members: np.ndarray
    source_members: np.ndarray
    target_mean: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    source_decoder: np.ndarray

    @classmethod
    def from_record(cls,record):
        k=len(record['target_members']);g=len(record['source_members']);d=record['hidden_dimension']
        return cls(np.asarray(record['target_members'],dtype=np.int64),np.asarray(record['source_members'],dtype=np.int64),
                   np.asarray(record['target_mean'],dtype=float),np.asarray(record['coefficient'],dtype=float).reshape(k,g),
                   np.asarray(record['intercept'],dtype=float),np.asarray(record['source_decoder'],dtype=float).reshape(g,d))

    def selected(self,codes,selected=False):
        x=np.asarray(codes,dtype=float)
        if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Finite two-dimensional target codes required')
        if not selected:x=x[:,self.target_members]
        if x.shape[1]!=len(self.target_members):raise ValueError('Target member count/order mismatch')
        return x

    def state(self,codes,selected=False):
        x=self.selected(codes,selected)
        return np.maximum((x-self.target_mean)@self.coefficient+self.intercept,0)

    def apply(self,recipient,donor=None,source_mask=None,selected=False,components=False):
        x0=self.selected(recipient,selected);a0=(x0-self.target_mean)@self.coefficient+self.intercept
        mask=np.ones(len(self.source_members)) if source_mask is None else np.asarray(source_mask,dtype=float)
        if mask.shape!=(len(self.source_members),) or not np.isfinite(mask).all():raise ValueError('Finite source mask in stored member order required')
        decoder=self.source_decoder*mask[:,None]
        if donor is None:
            delta=-np.maximum(a0,0)@decoder
            # Absolute baseline is the zero-target-code prediction, explicitly
            # retained rather than assigned to any selected member.
            left=self.intercept-self.target_mean@self.coefficient
            right=a0;dx=x0;sign=-1
            baseline=-np.maximum(left,0)@decoder
        else:
            x1=self.selected(donor,selected)
            if x1.shape!=x0.shape:raise ValueError('Recipient and donor shape mismatch')
            a1=(x1-self.target_mean)@self.coefficient+self.intercept
            delta=(np.maximum(a1,0)-np.maximum(a0,0))@decoder
            left=a0;right=a1;dx=x1-x0;sign=1
            baseline=np.zeros(decoder.shape[1])
        result=dict(delta=delta,source_state=np.maximum(a0,0))
        if components:
            difference=right-left
            q=np.divide(np.maximum(right,0)-np.maximum(left,0),difference,out=np.zeros_like(right),where=difference!=0)
            member_source=dx[:,:,None]*self.coefficient[None,:,:]*q[:,None,:]
            result.update(component_deltas=sign*np.einsum('nkg,gd->nkd',member_source,decoder),baseline_delta=baseline,secant=q)
        return result
