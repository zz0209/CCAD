# R011-F1 bounded matched causal gate v2 — signed paired components

Status: **LOCKED BEFORE V2 MODEL FORWARDS; SAME UNITS/SEQUENCES; AUDIT CLOSED**

V1 is immutable and remains a valid negative test of soft-marginal
interventions: `EUCLIDEAN_FCC_RELATION` qualified on 0/8 units, with median
next-state effect BCC `.02571` and normalized error `2.07417`.  That result
cannot be relabeled or deleted.

V1 also established that the marginal-only causalization discarded the signs
and paired rank components used by the frozen FCC fit.  The calibration BCC is
computed from `C_s A` and `C_t B`, whereas v1 intervened on positive row/column
marginals of `|A Sigma B^T|`.  The latter is a membership summary, not the
signed relation process.  It therefore cannot by itself adjudicate causal
validity of the full relation operator.

V2 changes only the causalization:

- use the already saved discovery-frozen signed loading columns `A[:,l]` and
  `B[:,l]` for every selected rank component;
- intervene on each paired component separately;
- use one block scale per method/side/sequence so the sum of component hook
  energies equals the primary source FCC block energy;
- aggregate source energy, target energy and cross-energy over components
  inside the original query-pair unit before BCC/error/coverage calculation;
- never sum factor columns into an arbitrary hook vector.

The aggregate quadratic hook quantities are invariant to a common signed
permutation of paired components.  Exact competing relations were already
refused by the rank-boundary gate; the single rank-2 unit is evaluated as a
two-component block, not as two independent samples.

Raw-hook PCA, SAE marginal PCA and stitching use their rank-matched projector
components.  Global FCC and matched random use signed rank-matched loading
components.  The native control uses the top-r frozen FCC membership atoms,
which gives it relation-informed target selection and is therefore a strong,
not disadvantaged, native baseline.  MSCC remains an explicit refusal.

All v1 unit identities, ranks and 16 sequence IDs are hash-bound by the v1
selection artifact.  The endpoint, numerical tolerances, qualification gates,
`.75` coverage gate and `.02` raw/global gain rules are unchanged.  No query,
candidate, rank, sequence, threshold, mean, or endpoint is selected from v1
causal outcomes.  Passing authorizes only the same full-640 pre-audit freeze
as v1; failure ends this signed-relation screen without opening audit.
