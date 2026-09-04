# R011-F2 / C045 bounded FCC estimator bracket

Status: **LOCKED BEFORE ALTERNATIVE CALIBRATION READ; AUDIT CLOSED**

The rejected fully-whitened contrastive CCA surface remains the reference.
Exactly two alternatives are allowed on the same 40 anchor queries, 120
collision-neighbor queries, source/target feature universes, source-only hard
negatives, ranks `{1,2,4,8}`, and `32 x <=128` feature-pair budget:

1. `ENERGY_BALANCED_PLS`: SVD of the positive-minus-negative raw
   cross-covariance, followed by unit positive discovery-energy normalization
   on each side and component.
2. `DIAGONAL_WHITENED_CORRELATION`: the same contrastive cross-covariance after
   per-feature energy normalization only; no dense within-side whitening.

Both emit signed paired loadings, cross operator, normalized absolute coupling
and overlapping marginals.  Candidate/query/negative ledgers cannot change.
There is no model causal forward in this bracket.

For each estimator and anchor ordered unit, select the first rank satisfying
all of: calibration BCC at least `.8`, normalized residual at most `.2`,
positive calibration hard-negative contrast, collision improvement over the
existing matched global relation at least `.05`, and rank-boundary relative
gap at least `.001`.  The estimator progresses only with coverage at least
10% of all 160 units, at least four energy strata, and nonzero coverage in all
20 represented ordered seed directions.

The `.8/.2` transfer thresholds inherit the frozen causal effect-consistency
tolerances; they are not fitted to the observed CCA failures.  If neither
alternative progresses, the current local contribution-kernel family closes.
No estimator suffix, threshold relaxation, full-640 expansion, causal forward,
or audit access is then permitted for this family.
