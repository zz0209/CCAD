"""Framework-neutral M0 contracts for hook capture, tokens, and SAE interventions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np


class ActivationContractError(ValueError):
    """Raised when a captured tensor or token record violates the frozen interface."""


@dataclass(frozen=True)
class HookPointContract:
    module_path: str
    layer_index: int
    tensor_kind: str
    hidden_size: int
    expected_rank: int = 3

    def validate_tensor(self, tensor: Any) -> None:
        if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
            raise ActivationContractError("hook output is not tensor-like")
        shape = tuple(int(value) for value in tensor.shape)
        if len(shape) != self.expected_rank:
            raise ActivationContractError(f"hook rank {len(shape)} != {self.expected_rank}")
        if shape[-1] != self.hidden_size:
            raise ActivationContractError(f"hidden size {shape[-1]} != {self.hidden_size}")


@dataclass(frozen=True)
class TokenAlignmentRecord:
    dataset_id: str
    document_id: str
    tokenizer_id: str
    tokenizer_revision: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_sha256: str


def build_token_alignment_record(
    *,
    dataset_id: str,
    document_id: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
) -> TokenAlignmentRecord:
    ids = tuple(_plain_int(value, "input_ids") for value in input_ids)
    mask = tuple(_plain_int(value, "attention_mask") for value in attention_mask)
    if not ids or len(ids) != len(mask):
        raise ActivationContractError("token IDs and attention mask must be nonempty and aligned")
    if any(value not in (0, 1) for value in mask):
        raise ActivationContractError("attention mask must be binary")
    payload = {
        "dataset_id": dataset_id,
        "document_id": document_id,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "input_ids": ids,
        "attention_mask": mask,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TokenAlignmentRecord(**payload, token_sha256=digest)


def assert_token_alignment(reference: TokenAlignmentRecord, candidate: TokenAlignmentRecord) -> None:
    if asdict(reference) != asdict(candidate):
        raise ActivationContractError("token alignment record mismatch")


def extract_primary_hook_tensor(output: Any, contract: HookPointContract) -> Any:
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    contract.validate_tensor(tensor)
    return tensor


def replace_primary_hook_tensor(output: Any, replacement: Any, contract: HookPointContract) -> Any:
    original = extract_primary_hook_tensor(output, contract)
    contract.validate_tensor(replacement)
    if tuple(original.shape) != tuple(replacement.shape):
        raise ActivationContractError("replacement shape differs from captured hook tensor")
    if str(original.dtype) != str(replacement.dtype):
        raise ActivationContractError("replacement dtype differs from captured hook tensor")
    original_device = getattr(original, "device", None)
    replacement_device = getattr(replacement, "device", None)
    if original_device is not None and str(original_device) != str(replacement_device):
        raise ActivationContractError("replacement device differs from captured hook tensor")
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    return replacement


def decoded_contribution(decoder: np.ndarray, codes: np.ndarray, feature_ids: Sequence[int] | None = None) -> np.ndarray:
    decoder = np.asarray(decoder)
    codes = np.asarray(codes)
    if decoder.ndim != 2 or codes.ndim < 2 or codes.shape[-1] != decoder.shape[1]:
        raise ActivationContractError("expected decoder[d,m] and codes[...,m]")
    ids = np.arange(decoder.shape[1]) if feature_ids is None else np.asarray(feature_ids, dtype=int)
    if ids.ndim != 1 or ids.size == 0 or np.any(ids < 0) or np.any(ids >= decoder.shape[1]):
        raise ActivationContractError("feature_ids must be a nonempty in-range vector")
    return np.einsum("...m,dm->...d", codes[..., ids], decoder[:, ids], optimize=True)


def sae_round_trip(
    hook_activation: np.ndarray,
    decoder_bias: np.ndarray,
    decoder: np.ndarray,
    codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hook = np.asarray(hook_activation)
    bias = np.asarray(decoder_bias)
    if bias.shape != (hook.shape[-1],):
        raise ActivationContractError("decoder bias must match hook hidden dimension")
    reconstruction = bias + decoded_contribution(decoder, codes)
    if reconstruction.shape != hook.shape:
        raise ActivationContractError("reconstruction shape differs from hook activation")
    return reconstruction, hook - reconstruction


def dynamic_group_ablation(
    hook_activation: np.ndarray,
    decoder: np.ndarray,
    codes: np.ndarray,
    feature_ids: Sequence[int],
) -> np.ndarray:
    hook = np.asarray(hook_activation)
    contribution = decoded_contribution(decoder, codes, feature_ids)
    if contribution.shape != hook.shape:
        raise ActivationContractError("group contribution shape differs from hook activation")
    return hook - contribution


def dynamic_group_swap(
    hook_activation: np.ndarray,
    decoder_left: np.ndarray,
    codes_left: np.ndarray,
    left_ids: Sequence[int],
    decoder_right: np.ndarray,
    codes_right: np.ndarray,
    right_ids: Sequence[int],
) -> np.ndarray:
    hook = np.asarray(hook_activation)
    left = decoded_contribution(decoder_left, codes_left, left_ids)
    right = decoded_contribution(decoder_right, codes_right, right_ids)
    if left.shape != hook.shape or right.shape != hook.shape:
        raise ActivationContractError("swap contributions must use the shared hook tensor unit")
    return hook - left + right


def _plain_int(value: Any, label: str) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ActivationContractError(f"{label} must contain integers")
    return int(value)
