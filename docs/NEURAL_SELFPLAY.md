# Neural search self-play: first closed-loop pilot

The supervised residual models now have an implemented search/self-play update
path. This follows the planning/generalization loop in
[Expert Iteration](https://arxiv.org/abs/1705.08439) and the search-policy/terminal-
value supervision used by [AlphaZero](https://arxiv.org/abs/1712.01815).
Their perfect-information strength or convergence claims do not transfer to our
approximate information-set search. This is a small development pilot; a running
job and a completed optimizer update are not evidence of improved playing strength.

## Why this experiment

The first completed 64-game CPU policy/value screen against the original teacher
scored 0W/2D/62L at actual 10+0.1, without technical errors or voids. All 2,940 moves
and actual-clock charges replayed correctly. The convolutional model with shallow
information-set search is currently much weaker than the native NNUE teacher.
Imitating teacher actions on existing replay does not itself establish an improved
search policy. Evidence: `results/20261026-neural-shared-vs-teacher.json` and its
independent replay audit. The remaining frozen phase screens continue unchanged.

## Actual search and terminal targets

The actor is the selected 9.34M-parameter shared residual checkpoint from the
completed 12-epoch pretraining. It searches both sides with 256 completed PUCT
simulations per decision, leaf batches of 32, and the inherited explicit chance
branches/root-information consistency. Root Dirichlet noise has alpha 0.3 and
weight 0.25. Move-sampling temperature is 1.0 for the first 30 plies and 0.1 thereafter.
Policy targets are the **unpruned actual child visit counts divided by their sum**;
they are distinct from sampled actions and from network priors. Every search must
finish all 256 backups. Terminal targets are the actual rule outcome from the
recorded actor's perspective. No teacher value is mixed into this update.

```math
\pi_t(a)=N_t(a)/\sum_bN_t(b),\qquad
L=-\sum_a\pi_t(a)\log p_\theta(a\mid o_t)+(v_\theta(o_t)-z_t)^2.
```

Each actual game has an environment RNG separate from search/exploration RNG.
Hidden identities are drawn uniformly without replacement on reveal/capture;
the search receives the acting observer's inventory and public rule state.
Real opponent-private ledgers are not used to construct simulated information.
Environment reveal/capture records are retained for post-game audits, rather than
being passed as the other player's observation. The compressed neural encoder
still omits unavailable history features; the search retains complete real history.

## Fixed pilot and independent replay

Sixteen games are fixed before generation, with 12 training games and 4 disjoint
checkpoint-selection games. Four independent inference workers share one GPU;
this is not distributed multi-GPU learning. A separate GPU update uses AdamW,
learning rate 1e-5, batch 128, two epochs, horizontal mirroring and gradient clipping 1.
The update begins from the same checkpoint that generated the episodes. A
minimum-validation checkpoint is selected after completed epochs; the initial
checkpoint is only a diagnostic. Four validation games cannot estimate strength.

Every episode stores observer bytes, integer root/child visits, selected moves,
environment reveal/capture events and actor-relative terminal values as numeric
arrays loaded with `allow_pickle=False`. A fresh seeded game replays every move,
checks both private ledgers, exact observation bytes, legal policy support,
completed visit totals and terminal-value perspective. Digests connect arrays,
episode metadata and the completed generation manifest. This audit verifies
record consistency, not that the approximate search is optimal.

The underlying self-play rules are pinned Yuan tiantian-v5, including its own
check/chase and repetition adjudication. They differ from the native-jqe external
benchmark; self-play terminal labels are not benchmark victories. A 1024-ply
resource cap aborts the episode and the fixed campaign; it never creates a draw
label or silently substitutes another game. Generation must complete before
training. Source, model, numerical weights and native acceleration are pinned.

Recipes: `results/neural-selfplay-pilot-generation-plan.json` and
`results/neural-selfplay-pilot-training-plan.json`. The pilot has no automatic
actor promotion. A learned gain requires fresh paired matches, adequate sample
size and replication; longer self-play or lower replay loss alone cannot supply it.

## Relationship to decreasing uncertainty

The trajectory replay separately counts genuinely fully public positions and
private dark captures. Those counts permit subsequent phase-specific value/policy
diagnostics without leaking private certainty into a public gate. The existing
public-phase value experts and dead-inventory canonicalization remain controlled
experiments. New current-policy visits/terminal values also permit testing whether
their previous negative supervised result came from teacher target bias. That
hypothesis remains unconfirmed; this pilot alone does not test it.

## Completed pilot and next fixed series

The pilot completed all 16 games: 945 positions, including 99 fully public positions
and 45 captor-private dark captures. Every position and policy target passed
complete seeded replay. Twelve games supplied 727 training positions; four
selection games supplied 218. Both GPU epochs completed 12 optimizer updates total.
This is an actual search/self-play/learning iteration, but deliberately small.
Selection loss decreased from its initial diagnostic; that is not strength
evidence. A separate two-game GPU-actor smoke lost both games, with all 115 legal
moves/clock charges independently replayed. Report and safe numeric raw episodes:
`results/20261026-neural-selfplay-pilot.json`. Models/gradients are FP32; NumPy's
normalization promotes soft targets/cross entropy to float64 in this pinned
implementation. This precision detail is preserved, rather than silently changing
the frozen pilot source or recipe.

Both pretrained CPU arms completed their 64-game teacher screens at 0W/2D/62L,
without technical failures; replay checked 2,940/3,038 moves respectively. This
near-floor performance cannot distinguish useful phase specialization. The fixed
phase-versus-shared comparison still runs and is not replaced by the new series.

`results/neural-selfplay-series-plan.json` fixes three successive research
iterations, each with 256 training games and 32 disjoint selection games, 256 PUCT
simulations per decision and four GPU epochs. These are successive updates of
one actor, not three independent training replicas. The first actor is the
original completed residual warm-start model, independently of the tiny pilot's
selected model. Each selected checkpoint becomes the next **research** actor
under the fixed validation recipe, even when a diagnostic match is negative.
This closes the on-policy collection loop that the earlier conservative NNUE
gates left at the unchanged teacher. It does not promote a public benchmark model.

Each iteration also completes two fresh 64-game development screens: against its
previous neural actor and the unchanged native teacher. Both neural actors use
the same GPU profile/leaf batch 32; the teacher is CPU NNUE. The teacher comparison
therefore measures the stated actual-clock resource profile, not equal total
compute. Match outcomes never select the next research actor or stop the series.
No 70%-win strong-baseline/report gate is claimed from these small screens.


The first larger generation/training stage has now completed 288 games and 15,682
positions, with four GPU epochs / 436 updates. Its previous-actor screen has completed at 24W/1D/39L (38.28% score; paired
95% bootstrap [28.125%,48.4375%]), providing no improvement evidence. All 3,160
moves and clocks replayed cleanly, without technical games or ambiguous margins.
The teacher screen has also completed at 1W/0D/63L, with 2,777 moves/clocks
cleanly replayed. The first full iteration is therefore complete, with no
improvement; the remaining two iterations are pending. The research actor
advances under the frozen recipe, and iteration 2 generation has begun. Exact episode/model hashes
and training curves: `results/20261026-neural-rl-series-first-training.json`.

First previous-actor evidence: `results/20261026-neural-rl-iter001-vs-previous.json`
and its independent `-audit.json`. The bounded series continues regardless of
these diagnostics; it does not promote a benchmark model.

Completed iteration 1 evidence: `results/20261026-neural-rl-iteration001.json`.
Both development screens share the stated GPU/CPU resource profiles; their
negative results do not establish behavior at longer clocks or larger training
scales.

[Search-cost diagnostics](NEURAL_SEARCH_COST.md) separate the existing default
lazy allocation from an eager counterfactual; they are not a new optimization
or playing-strength result.

Iteration 2 generation/training has also completed: 288 games, 19,590 positions,
four GPU epochs / 548 updates. Exact
curves and episode/checkpoint provenance are archived in
`results/20261026-neural-rl-series-second-training.json`. New-trajectory selection
losses are not comparable to iteration 1 as matched quality estimates.

Iteration 2 is now fully closed:17W/6D/41L against its previous actor and
0W/0D/64L against the native teacher, both technically clean;5,551moves/clocks
were independently replayed. Evidence: `results/20261026-neural-rl-iteration002.json`.
The frozen series has advanced to iteration3 without public model
promotion. A separate [normalization-state diagnosis](NORMALIZATION_DRIFT.md)
uses completed data; it does not alter that series.

A [fixed-statistics RL control](NORMALIZATION_DRIFT.md) bitwise reproduces the
original joint-update checkpoint in its ordinary-mode arm and tests a separate
frozen-BN update against fresh warm-model/teacher layouts. That update also
regressed; it does not alter the three-iteration series.

Iteration3 generation/training completed288games /21,068positions and four
GPU epochs /596updates. Epoch4 is selected on its existing selection split,
network SHA256 `40d1fde029ea18180f06098baf1f55e367a226f1601cd6185c18eda8c31c388d`.
Its fresh previous-actor screen is19W/13D/32L,39.84375%score, paired bootstrap95%
[28.125%,51.5625%], all3,893moves and clocks clean. This gives no improvement
evidence. The teacher screen is still live. Complete curves and provenance:
`results/20261026-neural-rl-series-third-training.json`; previous-match report
and audit: `results/20261026-neural-rl-iteration003-vs-previous{,-audit}.json`.

The separate [2048-simulation search-effort control](HIGH_SEARCH_RL.md) is running
from the original warm checkpoint, and a [teacher-static-value
ablation](TEACHER_LEAF_ABLATION.md) is queued to diagnose the weak neural search.
Neither changes the original series or promotes a research model.
