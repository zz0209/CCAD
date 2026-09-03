"""Core utilities for Cross-Seed Concept Alignment and Distillation."""

from .metrics import bcc_from_kernels, contribution_kernel, projector_subspace_consistency
from .synthetic import hadamard_gauge_instance, same_span_different_computation, same_sum_bloated_span

__all__ = [
    "bcc_from_kernels",
    "contribution_kernel",
    "projector_subspace_consistency",
    "hadamard_gauge_instance",
    "same_span_different_computation",
    "same_sum_bloated_span",
]
