"""Truth registry for labeled M1-NIP evaluation; forbidden to D0 runners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NIPTruth:
    family_id: str
    identification: str
    multiplicity: str | None
    minimum_supports: tuple[tuple[int, ...], ...]
    safety: str
    causal_outcome: str
    continuous_reference_feasible: bool = False
    full_group_portable: bool = False


def nip_truth(family_id: str) -> NIPTruth:
    rows = {
        "N01_structured_split": NIPTruth(family_id, "FOUND", "UNIQUE", ((0, 1),), "SAFE", "NOT_EVALUATED"),
        "N02_structured_merge_refactorization": NIPTruth(family_id, "FOUND", "UNIQUE", ((0, 1, 2),), "SAFE", "NOT_EVALUATED"),
        "N03_tied_native_supports": NIPTruth(family_id, "FOUND", "AMBIGUOUS", ((0, 1), (2, 3)), "SAFE", "NOT_EVALUATED"),
        "N04_absent_target": NIPTruth(family_id, "CERTIFIED_ABSENT", None, (), "NOT_APPLICABLE", "NOT_EVALUATED"),
        "N05_bloated_decoy": NIPTruth(family_id, "FOUND", "UNIQUE", ((0, 1),), "SAFE", "NOT_EVALUATED"),
        "N06_exact_dense_orthogonal_rotation": NIPTruth(family_id, "CERTIFIED_ABSENT", None, (), "NOT_APPLICABLE", "NOT_EVALUATED", full_group_portable=True),
        "N07_margin_separated_approximate_rotation": NIPTruth(family_id, "CERTIFIED_ABSENT", None, (), "NOT_APPLICABLE", "NOT_EVALUATED"),
        "N08_continuous_only_representation": NIPTruth(family_id, "CERTIFIED_ABSENT", None, (), "NOT_APPLICABLE", "NOT_EVALUATED", continuous_reference_feasible=True),
        "N09_cancellation": NIPTruth(family_id, "FOUND", "UNIQUE", ((0, 1),), "OBSERVATIONALLY_UNSAFE", "NOT_EVALUATED"),
        "N10_rare_occupancy": NIPTruth(family_id, "FOUND", "UNIQUE", ((0, 1),), "INSUFFICIENT_EVIDENCE", "NOT_EVALUATED"),
        "N11_downstream_cliff": NIPTruth(family_id, "FOUND", "UNIQUE", ((0,),), "SAFE", "CAUSAL_FAIL"),
        "N12_mean_mismatch": NIPTruth(family_id, "CERTIFIED_ABSENT", None, (), "NOT_APPLICABLE", "NOT_EVALUATED"),
    }
    try:
        return rows[family_id]
    except KeyError as exc:
        raise ValueError(f"unknown NIP truth family: {family_id}") from exc
