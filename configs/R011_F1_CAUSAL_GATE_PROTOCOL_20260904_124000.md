# R011-F1 bounded matched causal gate

Status: **LOCKED BEFORE CAUSAL MODEL FORWARDS; AUDIT CLOSED**

This gate consumes the discovery-frozen Euclidean FCC surface and the
calibration-only minimum-rank/refusal decisions.  It cannot refit a query,
candidate family, relation, rank, threshold, or hard negative.

## Unit and endpoint-blind selection

The independent descriptive unit is a source query and ordered seed pair.
Exactly one frozen `FOUND_RELATION` unit is selected from each of the eight
energy strata by the lowest immutable R009b selection hash, then target seed.
For each unit, the two calibration sequences with the largest source-query
squared-code energy are selected before any downstream endpoint is computed.
Sequence repeats are not independent samples.

The primary endpoint is the already-frozen next-layer residual state.  Next
logits are secondary.  The causal subset, sequences, methods, RMS matching,
and thresholds are all endpoint-blind.

## Relation-induced intervention

For a frozen FCC operator `K`, define `Pi=|K|/||K||_1`.  Its source and target
marginals weight the independently mean-centered native contribution banks:

`v_s(x)=sum_i Pi_s(i) (z_i^s(x)-mu_i^s) d_i^s`, and analogously for target.

This is the primary intervention because it is induced by the relation
operator and does not sum arbitrary CCA factor coordinates.  Every evaluated
control is independently rescaled on each sequence and side to the same hook
Frobenius norm as the source FCC intervention.  Rank means frozen relation or
projector rank; candidate and fitting budgets stay fixed.

## Controls in this bounded gate

- Euclidean FCC soft-marginal relation (primary);
- noncontrastive global FCC relation on the same feature universes and rank;
- source-query raw-hook conditional PCA;
- source-query SAE marginal PCA (the SCT endpoint);
- relaxed paired stitching/MAS-style bases;
- best functional single native atom;
- deterministic matched-rank random relation;
- the frozen MSCC refusal, reported as unavailable rather than a zero-effect
  intervention.

OT and Li15 remain mandatory for a full-640 freeze, but do not consume model
forwards in this early causal gate: failure against raw/global is already a
stopping condition and cannot be rescued by adding weaker controls.

## Frozen causal rules

A method qualifies on a unit when next-state source effect RMS is at least
`1e-6`, source-vs-target normalized effect error is at most `.20`, and effect
BCC is at least `.80`.  Primary coverage must be at least `.75` (six of eight).

For each of raw-hook and global-FCC controls, define:

- effect-consistency gain = control median normalized error minus primary
  median normalized error;
- query-specificity gain = control median source off-query fraction minus
  primary median source off-query fraction.

The primary relation must obtain at least `.02` gain on either axis against
each control.  This is an OR within each control, not across controls.  A pass
therefore cannot be caused solely by beating one easy baseline.  Numerical
conformance additionally requires no-op max error at most `1e-6` and cached
raw-hook replay relative RMS at most `1e-4` (absolute max at most `1e-3`).

## Progression

Failure produces a bounded pre-audit negative and returns the eight screened
relations to `UNRESOLVED_RELATION: CAUSAL_SPECIFICITY_GATE_FAILED`.  It does
not change the FCC mother problem.  Passing only authorizes one full-640
pre-audit relation/baseline freeze.  It is not C1-FCC or C2-FCC evidence and
does not open audit.

The original v1 language requiring a gain over “Euclidean FCC” applied to a
C040-metric primary lane.  C040 was prospectively rejected, so the surviving
Euclidean FCC lane is not compared with itself; it must beat both raw-hook and
global-FCC controls under the rule above.
