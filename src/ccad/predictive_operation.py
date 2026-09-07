"""Apply an exported, signed source-operation predictor to target SAE codes.

This consumer fits nothing and needs no SAE weights or language-model runtime.
Its output is a residual-stream update, not a native feature-zeroing operation.
The caller must use the model, hook, target SAE and code scale in the metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PredictiveOperation:
    target_members: np.ndarray
    predictive_vectors: np.ndarray
    latent_width: int
    metadata: dict

    def __post_init__(self):
        ids = np.asarray(self.target_members)
        vectors = np.asarray(self.predictive_vectors, dtype=np.float64)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('target_members must be a one-dimensional integer array')
        if self.latent_width < 1 or len(ids) != len(set(ids.tolist())):
            raise ValueError('latent_width must be positive and target members unique')
        if np.any(ids < 0) or np.any(ids >= self.latent_width):
            raise ValueError('target member is outside the declared SAE width')
        if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] < 1:
            raise ValueError('predictive_vectors must have shape (members, hook dimensions)')
        if not np.all(np.isfinite(vectors)):
            raise ValueError('predictive vectors must be finite')
        ids = ids.astype(np.int64, copy=True); vectors = vectors.copy()
        ids.setflags(write=False); vectors.setflags(write=False)
        object.__setattr__(self, 'target_members', ids)
        object.__setattr__(self, 'predictive_vectors', vectors)
        object.__setattr__(self, 'metadata', dict(self.metadata))

    def predict(self, code_difference):
        """Return delta h for (..., full SAE width) donor-minus-recipient codes."""
        delta = self._array(code_difference, self.latent_width)
        return self.predict_selected(delta[..., self.target_members])

    def predict_selected(self, selected_difference):
        """Return delta h; columns must follow target_members exactly."""
        delta = self._array(selected_difference, len(self.target_members))
        return delta @ self.predictive_vectors

    def component_deltas(self, selected_difference):
        """Return (..., members, hook dimensions); use only for small case sets."""
        delta = self._array(selected_difference, len(self.target_members))
        return delta[..., :, None] * self.predictive_vectors

    @staticmethod
    def _array(value, width):
        value = np.asarray(value, dtype=np.float64)
        if value.ndim < 1 or value.shape[-1] != width:
            raise ValueError(f'Expected code dimension {width}; got {value.shape}')
        if not np.all(np.isfinite(value)):
            raise ValueError('Code differences must be finite')
        return value

    def save(self, path):
        """Write a NumPy-only bundle without overwriting an existing result."""
        with Path(path).open('xb') as handle:
            np.savez_compressed(handle, schema_version=np.array('ccad.predictive_operation.v1'),
                                target_members=self.target_members,
                                predictive_vectors=self.predictive_vectors,
                                latent_width=np.array(self.latent_width, dtype=np.int64),
                                metadata_json=np.array(json.dumps(self.metadata, sort_keys=True)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as bundle:
            if str(bundle['schema_version'].item()) != 'ccad.predictive_operation.v1':
                raise ValueError('Unsupported predictive operation schema')
            return cls(bundle['target_members'], bundle['predictive_vectors'],
                       int(bundle['latent_width'].item()), json.loads(str(bundle['metadata_json'].item())))
