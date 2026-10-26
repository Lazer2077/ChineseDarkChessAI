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
positions, with four GPU epochs / 436 updates. Its previous-actor and teacher
matches are separate pending diagnostics, so neither the first full iteration
nor the three-iteration series is reported complete. Exact episode/model hashes
and training curves: `results/20261026-neural-rl-series-first-training.json`.
