"""Minimal local adapters of RAVEL's MIT DBM and low-rank DAS operators.

Original: explanare/ravel e421fc9e132b3613b1d0020f856d539217fb30ca,
src/methods/{differential_binary_masking,distributed_alignment_search}.py.
Retained formulas: train sigmoid/temperature and binary evaluation for DBM;
torch orthogonal parametrization and delta W^T W for DAS. Training and data
interfaces here are CCAD's, not the complete original RAVEL benchmark.
Original copyright/license is retained with the vendored source in artifacts.

MIT License. Copyright (c) 2024 Jing Huang.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations
import torch
from torch import nn
from ccad.semantic_participation import union_participation,project_sparse_gates


class SemanticNative(nn.Module):
    def __init__(self,width,attributes,budget,exclusive=False):
        super().__init__();self.gates=nn.Parameter(torch.zeros(width,attributes))
        self.budget,self.exclusive=budget,exclusive

    def forward(self,difference,decoder,control):
        return (difference*union_participation(self.gates,control))@decoder

    def project(self):project_sparse_gates(self.gates,self.budget,self.exclusive)


class MultiDBM(nn.Module):
    def __init__(self,dim,attributes):
        super().__init__();self.mask=nn.Parameter(torch.zeros(dim,attributes));self.temperature=.01

    def forward(self,difference,decoder,control):
        gates=torch.sigmoid(self.mask/self.temperature) if self.training else (self.mask>0).to(self.mask.dtype)
        return difference*union_participation(gates,control)


class MultiDAS(nn.Module):
    def __init__(self,dim,attributes,rank):
        super().__init__();self.rotations=nn.ModuleList([
            nn.utils.parametrizations.orthogonal(nn.Linear(dim,rank,bias=False)) for _ in range(attributes)])

    def forward(self,difference,decoder,control):
        # Single controls are exactly the original DAS operation. Multiple
        # controls use stated ordered sequential projections toward fixed donor.
        delta=torch.zeros_like(difference)
        for a,layer in enumerate(self.rotations):
            if float(control[a])!=0:
                delta=delta+control[a]*((difference-delta)@layer.weight.T)@layer.weight
        return delta
