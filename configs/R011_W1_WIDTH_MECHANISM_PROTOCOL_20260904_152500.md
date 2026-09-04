# R011-W1 / C049 width-only training mechanism screen

Status: **LOCKED BEFORE NEW TRAINING; AUDIT CLOSED**

The treatment changes only `num_latents` from 3,072 to 16,384. Pythia-160M-deduped revision, layer5 resid-post hook, sparsify commit, TopK k32, the 4,194,304-token FineWeb asset/order, optimizer, learning rate, warmup, batch, validation, seeds1/2 and float32 remain identical to R011-NR1 k32. Width16,384 is one literature-anchored level, not a sweep; the single-seed public width65,536 checkpoint remains context only.

The independent unit is an initialized SAE seed. Seeds1/2 form a paired mechanism screen blocked against the existing width3072/k32 seeds; tokens and sequences are repeated measurements. This screen can decide whether width merits further evidence but cannot establish a population claim.

First run one seed1 256-step capacity smoke. Stop before formal training if peak allocated VRAM exceeds 6 GiB, projected time exceeds 50 minutes per seed, projected exact plus safe checkpoints exceed 1 GiB per seed, L0 is not exactly32, or hook/CE pipeline checks fail. Estimated encoder plus decoder size is 25,165,824 fp32 weights (about96 MiB before gradients/optimizer). The smoke is engineering evidence only.

If capacity passes, train exactly seeds1/2 sequentially. Each formal run must pass the existing12 identity/artifact checks, FVE>=.97, CE recovered>=.93, alive fraction>=.50, exact L0=32 and decoder norm error<=1e-5. Only then build paired codes and run the same full-dictionary PW-MCC, frequency-stratified best-single BCC and source-evaluable native coverage ledger. FCC may follow only after this ledger and retains the existing `.8/.2`, control and refusal semantics.

Width is useful only if two-seed PW-MCC improves by at least .05 over the matched width3072 baseline .48940 and complete-query native FOUND coverage is at least10% overall and5% in each direction. Best-single BCC cannot substitute. Failure closes this width mechanism at4.19M tokens. No extra seed, width, k, hook, budget, optimizer, causal forward, full audit, five-seed promotion or longer continuation is authorized here.
