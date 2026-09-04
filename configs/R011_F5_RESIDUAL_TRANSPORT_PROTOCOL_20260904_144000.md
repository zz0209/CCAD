# R011-F5 / C048 shared-nuisance residual FCC transport

Status: **LOCKED BEFORE REAL CALIBRATION READ; AUDIT CLOSED**

C048 tests one new representation and no parameter sweep.  A shared nuisance
projector is fitted from 4,096 document-balanced, query-agnostic discovery raw
hook states, centered only with the independent mean split.  Its rank is the
unique smallest prefix explaining at least 90% of discovery global variance,
with a hard maximum of 64; failure to reach 90% by rank 64 stops the setting.
No query, target seed, calibration value, or endpoint participates in this
projector or its rank.

The same frozen projector residualizes all three hook-space processes:
source-local SAE contribution, complete target-SAE reconstruction, and raw
hook.  Source query construction then follows C047 on the residual source
process; ridge reduced-rank transport remains `.001` with ranks `{1,2,4,8}`.
Each query must retain at least 20% of its pre-residual discovery source-process
energy or it is refused, not rescued by changing nuisance rank.

The independent unit remains source query by ordered seed pair; tokens are
repeated measurements and seed direction plus the eight existing energy strata
are blocks.  Matched controls are residualized query-conditioned raw-hook
transport, residualized query-agnostic whole-SAE/global transport, and the
frozen unresidualized C047 surface.  All share rank, ridge, observation budget,
source query, and calibration evaluation.

Meaningful transfer remains BCC at least `.8`, source-normalized residual at
most `.2`, positive hard-negative specificity, collision improvement at least
`.05`, rank gap at least `.001`, and specificity at least `.05` above the best
matched control.  Progression requires at least 10% of 160 ordered units, at
least four strata, and all 20 directions.  Synthetic gates must first verify
the unique nuisance-rank rule, global-nuisance recovery, orthogonal query-signal
preservation, nuisance-only refusal, and explicit rank-deficiency refusal.

Failure stops residual FCC and moves the next evidence unit to SAE training or
configuration rather than another residualization/rank variant.  No causal
forward, full-640 expansion, audit read, or C1/C2-FCC claim is allowed here.
