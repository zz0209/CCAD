# R011-F4 / C047 query-conditioned hook-space reduced-rank transport

Status: **LOCKED BEFORE REAL CALIBRATION READ; AUDIT CLOSED**

This is a fresh FCC representation setting, not another native-coordinate
estimator or candidate-cap sweep.  The source query remains target-blind and
source-only.  For each candidate rank, weighted PCA on the discovery-positive
sum of its 32-feature centered source contribution family freezes a source
query projector; the source dynamic process is that local reconstruction
projected into this source-only subspace.  The target input is the complete
centered reconstruction from the target SAE in the same 768-dimensional hook
space.  A ridge
reduced-rank-regression map is fit on discovery query-positive observations and
then frozen.  Candidate ranks are exactly `{1,2,4,8}` and the ridge is fixed at
`1e-3` times the mean nonzero target-process energy; calibration may select only
the minimum passing rank or refuse.

The independent unit remains source query by ordered seed pair.  Tokens are
repeated measurements inside the unit; source seed, target seed, and the eight
pre-existing energy strata are blocking factors.  The same 40 anchors and 120
source-only collision neighbors are used.  Means remain frozen from the mean
split.  No query, row, rank, threshold, or control may be changed after seeing
calibration.

Meaningful transfer retains BCC at least `.8`, source-normalized residual at
most `.2`, positive-minus-hard-negative BCC greater than zero, collision
improvement over the matched global transport at least `.05`, and rank-boundary
gap at least `.001`.  In addition, query specificity must exceed the better of
two matched controls by at least `.05`: (1) query-conditioned raw-hook transport
and (2) query-agnostic whole-SAE/global transport.  Controls use the same rank,
ridge rule, source projector, observation budget, energy matching, and
calibration evaluation.
This makes raw/global collapse a refusal, not a success.

Progression requires at least 10% of all 160 anchor ordered units, at least four
energy strata, and all 20 ordered seed directions.  Before any real calibration
read, deterministic synthetic fixtures must pass rotation, split/merge,
query-null/global-nuisance refusal, raw-control rejection, and explicit
rank-deficiency refusal.  A failed real screen stops this representation: no
new ridge/rank grid, causal forward, full-640 expansion, audit read, or claim of
C1/C2-FCC is allowed.
