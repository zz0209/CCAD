"""Composition-compatible source controls and a portable signed readout.

Shared orthogonal blocks and independent DAS projections are classical
operator choices. This module does not assign semantic labels to features.
"""
from __future__ import annotations
import torch
from torch import nn
from ccad.ravel_controls import MultiDAS
from ccad.semantic_readout import apply_numpy


class SharedDAS(nn.Module):
    def __init__(self, dim, attributes, rank):
        super().__init__()
        if attributes*rank>dim:raise ValueError('Disjoint blocks exceed hook dimension')
        self.attributes,self.rank=attributes,rank
        self.rotation=nn.utils.parametrizations.orthogonal(nn.Linear(dim,attributes*rank,bias=False))

    def bases(self):return self.rotation.weight.reshape(self.attributes,self.rank,-1)

    def forward(self,difference,decoder,control):
        weight=self.rotation.weight
        strengths=control.repeat_interleave(self.rank)
        return ((difference@weight.T)*strengths)@weight


def make_family(dim,attributes,rank,common):
    return SharedDAS(dim,attributes,rank) if common else MultiDAS(dim,attributes,rank)


def get_bases(operator):
    if isinstance(operator,SharedDAS):return operator.bases()
    return torch.stack([layer.weight for layer in operator.rotations])


def apply_torch(difference,bases,control,reverse=False):
    delta=torch.zeros_like(difference)
    for a in (reversed(range(len(bases))) if reverse else range(len(bases))):
        if float(control[a])!=0:
            weight=bases[a]
            delta=delta+control[a]*((difference-delta)@weight.T)@weight
    return delta

