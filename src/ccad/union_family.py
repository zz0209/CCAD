"""Finite control-cube representation of overlapping native operation families.

Standard multilinear interpolation supplies the identity. Its empirical value
must be tested; no semantic or downstream KL guarantee follows by itself.
"""


def binary_controls(attributes, *, device=None, dtype=None):
    """All cube vertices in bit-mask order, including the exact zero edit."""
    import torch
    indices=torch.arange(2**attributes,device=device)
    return ((indices[:,None]>>torch.arange(attributes,device=device))&1).to(dtype or torch.float32)


def vertex_weights(controls):
    """Convex interpolation weights for controls in the unit cube.

    For fixed donor/base states, every union edit is multilinear in controls.
    Thus its hook-space difference across two dictionaries is the same convex
    mixture of vertex differences. This identity does not interpolate nonlinear
    LM output distributions, semantic accuracy, or KL divergence.
    """
    import torch
    vertices=binary_controls(controls.shape[-1],device=controls.device,dtype=controls.dtype)
    return torch.prod(torch.where(vertices.bool(),controls[...,None,:],1-controls[...,None,:]),dim=-1)
