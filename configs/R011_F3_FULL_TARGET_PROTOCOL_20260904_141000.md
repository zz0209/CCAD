# R011-F3 / C046 full-target scalable FCC screen

Status: **LOCKED BEFORE FULL-TARGET CALIBRATION READ; AUDIT CLOSED**

This screen tests candidate truncation once.  It retains the frozen 32-feature
source families, 40 anchor and 120 collision-neighbor queries, source-only
positive/hard-negative ledgers, ranks `{1,2,4,8}`, and energy-balanced
contrastive PLS.  The target universe is exactly all 3,072 native features.
There is no intermediate target cap or target-size sweep.

The implementation may materialize a `32 x 3072` cross-covariance but must not
materialize a `3072 x 3072` target Gram.  Target loading energy is computed by
streaming selected rank-component contributions through sparse codes and the
decoder.  A matched full-target, noncontrastive global PLS relation supplies
the collision control.  Candidate/search budgets are reported as 98,304
feature pairs versus at most 4,096 in the capped reference.

Discovery fits the relation.  Calibration evaluates frozen loadings.  The
minimum-rank rule is unchanged: BCC at least `.8`, normalized residual at most
`.2`, positive hard-negative contrast, collision improvement over matched
full-target global relation at least `.05`, and rank-boundary gap at least
`.001`.  Progression additionally requires at least 10% of all 160 anchor
ordered units, at least four strata, and all 20 ordered directions.

No model causal forward, full-640 expansion, or audit access is allowed.  If
the full-target setting fails, candidate truncation is closed as the
explanation for the current SAE configuration; the next work must change the
FCC representation/configuration setting rather than add another cap,
estimator, or weaker threshold.
