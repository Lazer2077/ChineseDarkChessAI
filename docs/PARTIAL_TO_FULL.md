# Exploiting the transition from hidden to public state

Research direction added from the user's observation: Jieqi progressively loses
future-relevant hidden identities and eventually has a fully public continuation.
This should influence computation and learning, rather than using move number
as a proxy for information. Existing fixed experiments remain unchanged.

## Three different notions of certainty

1. The number of on-board face-down pieces is public and never increases under
   the standard rules. It is a useful, but incomplete, progress measure.
2. An actor may infer a remaining identity from inventory before it is revealed.
   That actor's certainty need not be shared by the opponent.
3. When no on-board hidden identity remains, unseen historical capture identities
   no longer affect future legal moves, reveals or payoffs. They need not all be
   disclosed for the continuation game graph to be fully public. Public repetition,
   check/chase history and counters still matter and must be retained.

A public inventory can sometimes determine all remaining identities while some
pieces are still face down. They retain their original-position movement roles
until first moved; inference does not permit replacing them with revealed pieces.
Even after every piece is revealed, Jieqi's mobile advisors/elephants do not turn
into ordinary Xiangqi rules. A deterministic continuation solver must retain the
actual Jieqi rules.

## A measurable information variable

For one colour, let c be its visible-inventory count vector, including unknown
off-board removals, and n the number of on-board hidden squares. Under the current
exchangeable, without-replacement urn prior, define H_n(c) as joint Shannon entropy
of the ordered identities on those n squares. It accounts for correlated pieces:

```math
H_0(c)=0,\qquad
H_n(c)=H(\mathrm{Cat}(c/|c|))+
\sum_i\frac{c_i}{|c|}H_{n-1}(c-e_i).
```

Entropy sums over the two independent colour bags. Initial uncertainty is
56.6865 bits, counting types rather than distinguishable copies of the same type.
This is an inventory-prior measure, not a strategy-conditioned posterior or a
claim that entropy must decrease after every possible signal/model update.
The decomposition into public state and a smaller hidden component is related
to [mixed observability models](https://www.comp.nus.edu.sg/~leews/publications/rss09.pdf);
the two-player game still needs its own strategic assumptions.

The extractor distinguishes actor-private inventory and public inventory. For
the latter, each actor's own-colour urn contains only public reveal deductions;
the other actor's private capture deductions are never used as public knowledge.
Whole-game boundaries and public move/reveal transitions certify reconstruction.
Singleton trajectories with insufficient evidence remain explicitly unavailable.
Tests check exact small-bag enumeration, without-replacement correlations and
invariance of public inventory to captor-private counts.

## Measurements on the existing fixed dataset

The first 8,192-game dataset has 607,652 raw / 405,959 retained positions:

- 64,389 raw / 41,369 retained positions have no hidden on-board pieces.
- Another 13,215 raw / 8,182 retained positions have zero public-prior identity
  entropy despite remaining face-down pieces.
- 22,178 raw / 13,357 retained positions have actor-only certainty while public
  inventory remains uncertain. An actor-zero gate alone would switch too early
  if treated as a common-information perfect-information game.

The fixed validation set has 3,061 retained fully public positions, 269 more
with public-inferred certainty despite face-down pieces, and 679 with actor-only
certainty. No entropy rises occurred in the 591,271 measured same-observer
training transitions under this particular urn model; this is descriptive
sample evidence, not a general POMDP monotonicity theorem.

There are 2,893 retained full-public validation positions with residual off-board
inventory features. Holding the board fixed and clearing only those features in
the original teacher changes its evaluation by mean absolute 98.64 internal score
units, median 78.90, maximum 463.68. This diagnoses representation dependence;
zeroed inputs are a representation intervention and may be outside the original
training distribution. It does not prove that the intervention improves play.
A fixed opponent policy may depend on past private information even though the
optimal fully public continuation value need not depend on dead inventory.
Evidence: results/20261026-observability-analysis.json.

## First implemented ablation: forget irrelevant belief features

The native backbone already stops branching on reveals when there are no dark
pieces. We do not claim that inherited behavior as a new algorithm.

Our new variant omits NNUE inventory features only when **both on-board dark
counts are zero**. It retains the actual Position inventory, all game history,
rule counters, legal moves and chance search; no still-dark piece is exposed.
The feature transformer refreshes once when the last dark piece is revealed or
captured, preventing the old incremental accumulator from retaining dead features.
Thereafter ordinary incremental updates remain available. Float and quantized
paths implement the same abstraction.

Matching training data zeroes only inventory input columns at those fully public
positions. Search scores, outcomes, flags and other data are unchanged. The small
network has completed 15 GPU epochs with the original training recipe; a better
validation loss or playing strength is not assumed. Two 64-game development arms
use the same new layout seed, actual clock 5+0.05 and one thread/64 MB:

- B versus A: inference canonicalization with the unchanged trained control's weights.
- C versus A: canonicalization plus matched canonical-data training.

A is the original feature/search version with the identically trained raw-data
small model. This separates feature intervention from retraining. Both arms are
frozen before outcomes, with no tuning between arms or pooling into a strength
claim. Native checks exercised 32 full-public invariances, 76 last-dark transitions
and 5,218 ordinary updates in float and quantized modes. These are correctness
checks, not wins. Recipe: results/observable-endgame-plan.json.

## Further hypotheses, not implemented claims

Use public/actor uncertainty to allocate sampling versus deterministic search,
with matched wall-clock controls against the existing chance-search baseline.
Train phase-conditioned experts or a shared network with an information-dependent
gate, while checking for discontinuities at the boundary. Use accurate deterministic
continuation targets to bootstrap earlier partial-observation states, retaining the
same observer's history and measuring teacher bias. Endgame curricula must be
compared with equal-data/equal-compute controls; lower endgame prediction error
alone does not establish stronger whole-game play.

## First inference-only screen completed

B versus A completed all 64 games at 29W/6D/29L, exactly 50% score, paired
bootstrap 95% [0.40625, 0.6015625], without technical failures or voids.
Canonicalization establishes the intended full-public input invariance but
this screen does not establish a strength improvement. The separately frozen retrained C-versus-A arm also completed: 30W/7D/27L,
52.34375%, paired bootstrap 95% [0.421875, 0.625], without technical failures
or voids. No tuning occurred between arms. Neither arm establishes a strength
benefit; these small whole-game screens also do not rule out phase-specific
benefits. A future held-out full-public start-position study should preserve
rule-relevant history and use matched data and compute. Evidence:
results/20261026-observable-endgame-B.json and
results/20261026-observable-endgame-C.json.

Stage-specific follow-up diagnostics and their game-level sampling limits are
in [PHASE_DIAGNOSTICS.md](PHASE_DIAGNOSTICS.md). Only 88 of 512 validation games
contribute retained fully public positions; counts of positions are not counts
of independent games.
