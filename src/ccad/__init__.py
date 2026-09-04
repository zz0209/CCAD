"""Core utilities for Cross-Seed Concept Alignment and Distillation."""

from .metrics import bcc_from_kernels, contribution_kernel, projector_subspace_consistency
from .subspace_transport import direct_process_transfer_metrics, fit_weighted_pca, fit_weighted_stitching, select_weighted_support, transfer_metrics
from .synthetic import hadamard_gauge_instance, same_span_different_computation, same_sum_bloated_span

__all__ = [
    "bcc_from_kernels",
    "contribution_kernel",
    "projector_subspace_consistency",
    "fit_weighted_pca",
    "fit_weighted_stitching",
    "direct_process_transfer_metrics",
    "select_weighted_support",
    "transfer_metrics",
    "hadamard_gauge_instance",
    "same_span_different_computation",
    "same_sum_bloated_span",
]
