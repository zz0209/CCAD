"""Deterministic, query-agnostic design helpers for C040 hook probes."""
from __future__ import annotations

import hashlib

import numpy as np


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def select_document_balanced_states(
    sequence_records: list[dict],
    *,
    split: str,
    count: int,
    token_positions: tuple[int, ...],
    salt: str,
) -> list[dict]:
    """Select content-hashed states while limiting document concentration.

    Sequence records can straddle document boundaries without token-level
    provenance.  A state is therefore blocked by every document listed for its
    sequence.  Round-robin selection first gives each document an opportunity
    to contribute one unused state, then a second, and so on.
    """

    if count <= 0 or not token_positions:
        raise ValueError("count and token_positions must be positive")
    records = [record for record in sequence_records if record.get("split") == split]
    if not records:
        raise ValueError(f"no sequence records for split {split!r}")
    by_document: dict[str, list[dict]] = {}
    for record in records:
        sequence_index = int(record["sequence_index"])
        token_hash = str(record["token_sha256"])
        documents = tuple(sorted(str(value) for value in record["document_ids"]))
        if not documents:
            raise ValueError("sequence record has no document IDs")
        for position in token_positions:
            state_key = f"{split}:{sequence_index}:{position}:{token_hash}"
            state_hash = hashlib.sha256(f"{salt}|{state_key}".encode()).hexdigest()
            state = {
                "split": split,
                "sequence_index": sequence_index,
                "token_position": int(position),
                "token_sha256": token_hash,
                "document_ids": list(documents),
                "state_key": state_key,
                "selection_hash": state_hash,
            }
            for document in documents:
                by_document.setdefault(document, []).append(state)
    for candidates in by_document.values():
        candidates.sort(key=lambda row: (row["selection_hash"], row["sequence_index"], row["token_position"]))
    documents = sorted(by_document, key=lambda value: hashlib.sha256(f"{salt}|doc|{value}".encode()).hexdigest())
    selected: list[dict] = []
    used: set[str] = set()
    depth = 0
    while len(selected) < count:
        added = 0
        for document in documents:
            candidates = by_document[document]
            available = next((row for row in candidates if row["state_key"] not in used), None)
            if available is not None:
                row = dict(available)
                row["blocking_document_id"] = document
                row["document_round"] = depth
                selected.append(row)
                used.add(row["state_key"])
                added += 1
                if len(selected) == count:
                    break
        if added == 0:
            raise ValueError("insufficient distinct states for requested balanced sample")
        depth += 1
    return selected


def hashed_vocab_sketch(vocab_size: int, count: int, salt: str) -> tuple[np.ndarray, np.ndarray]:
    """Choose unique vocabulary coordinates and fixed signs by content hash."""

    if vocab_size <= 0 or count <= 0 or count > vocab_size:
        raise ValueError("invalid vocabulary sketch dimensions")
    ranked = sorted(
        range(vocab_size),
        key=lambda token: hashlib.sha256(f"{salt}|vocab|{token}".encode()).digest(),
    )[:count]
    signs = [1.0 if hashlib.sha256(f"{salt}|sign|{token}".encode()).digest()[0] & 1 else -1.0 for token in ranked]
    return np.asarray(ranked, dtype=np.int64), np.asarray(signs, dtype=np.float64)


def rademacher_direction(hidden_size: int, state_key: str, direction_index: int, salt: str) -> np.ndarray:
    """Return a deterministic unit-norm Rademacher probe direction."""

    if hidden_size <= 0 or direction_index < 0:
        raise ValueError("invalid direction dimensions")
    rng = np.random.default_rng(stable_seed(salt, state_key, direction_index))
    values = rng.integers(0, 2, size=hidden_size, dtype=np.int8).astype(np.float64)
    values = (2.0 * values - 1.0) / np.sqrt(hidden_size)
    return values
