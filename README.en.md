# Chinese Dark Chess AI

[简体中文](README.md) | **English**

Research on imperfect-information search and self-play learning for **full-board Jieqi**. We aim to outperform publicly available, runnable, independent Jieqi engines in reproducible experiments with compatible rules. The project is licensed under GPL-3.0; external engines, models, and rules are attributed separately.

**Three pinned opponent versions have completed both confirmation sets under fixed rules and actual clocks. This does not establish an improvement over the upstream teacher; broader coverage remains incomplete.** The project focuses on algorithms and experimental evidence. Half-board dark chess (Banqi) is outside its scope.

## Problem: hidden pieces and private information

Jieqi is more precisely a two-player zero-sum partially observable stochastic game. Fixing the opponent's policy allows one player's decision problem to be modeled as a POMDP; optimal single-agent planning does not directly imply an equilibrium guarantee for the two-player game.

A player's information history consists of the public board, reveals and move history, and the identities of hidden pieces they have captured. A hidden capture reveals the identity only to the capturing player. Pieces captured by the opponent whose identities remain unknown to the player must stay in that player's unknown inventory; the simulator's true state must not be used to exclude them. Hidden identities are assigned **without replacement**, so piece identities are correlated.

The current prototype and adapters separate true state from player observations, with tests for invariance to unobserved identities, private capture visibility, and reconstruction from complete histories. The inventory-based belief does not fully model how the opponent's actions reveal their private information. This remains a research gap, and the belief should not be described as an exact posterior.


## Current method: search-guided value learning

The main experiments use a pinned MIT-licensed version of [devinokeefe/jieqi](https://github.com/devinokeefe/jieqi): alpha-beta/PVS search, approximations to reveal chance branches, and an NNUE value network. The native search architecture, trainer, and initial teacher weights are upstream contributions; see [provenance](docs/PROVENANCE.md). The current control model has 387,089 parameters.

The learning loop is: self-play guided by the current model → collect search scores and terminal outcomes → replay the latest two generations → update the value network on a GPU → run an independent paired match → update the next generation's actor only if the acceptance gate passes. Resource-truncated or erroneous games do not enter this training set. Rejected candidates are recorded, and the previous actor is retained.

For a search score s and terminal outcome z∈{−1,0,1}, both from the acting player's perspective, the training target is:

```math
y = \lambda\,\sigma(s/K) + (1-\lambda)\,(z+1)/2,\qquad
\mathcal L = \mathbb E[(\sigma(V_\theta(o)/K)-y)^2].
```

The current series uses λ=0.7 and K=430. This is value-based expert iteration combining search-target distillation with terminal-outcome learning. A separate [residual policy/value plus information-set PUCT self-play RL loop](docs/NEURAL_SELFPLAY.md) now updates its current actor from real search visits and terminal rewards: a 16-game pilot completed, and three successive learning iterations are running. Iteration 1 scored 24W/1D/39L versus its previous actor and 1W/0D/63L versus the original teacher, both technically clean. No learned strength gain is established, and full POMCP, CFR and ReBeL are not implemented. The early 15-dimensional linear-policy prototype checks observation isolation and the search/training pipeline; it does not represent the strength of the current main experiment.

## Experimental scale and observed results

The following is an evidence snapshot dated 2026-10-26. Running jobs are not completed experiments.

[Completed version-specific confirmations](docs/CONFIRMED_RESULTS.md), each set with 200 games / 100 color-swapped layouts and actual clocks of 5+0.05:

| Pinned opponent | Set A (W/D/L) | Set B (W/D/L) |
| --- | --- | --- |
| AB-JChess v0.2d | 162/7/31, 82.75% | 150/2/48, 75.5% |
| universe-st/Jieqi `e00b667c` | 200/0/0, 100% | 198/2/0, 99.5% |
| brianhliou/Pika WASM branch `bcc83f88`, native build | 181/1/18, 90.75% | 179/3/18, 90.25% |

All six sets have no technical failures or void attempts, with both paired bootstrap and conservative Hoeffding 95% score lower bounds above 50%. All 65,826 moves and clock records were replay-audited. Results apply only to these versions and the documented referee/adapter profiles.

- The [three-seed NNUE series](results/20261026-continual-three-seeds.json) completed two iterations per seed, 8,192 games per iteration: **49,152 games and 3,618,389 raw searched positions**, with zero rejected generation games. Each iteration trained for 15 GPU epochs. The six 200-game internal gate scores were 46.75%, 47.75%, 50.0%, 55.75%, 50.75% and 46.5%; none passed. All generation therefore remained guided by the original teacher, without a successfully improved behavior policy.
- NNUE training uses a GPU and self-play search mainly uses CPUs. Two separate GPUs also completed independent 9.34M/10.07M policy/value pretraining arms, 12 epochs and 56,652 updates each. This is supervised teacher pretraining, not distributed multi-GPU training or an AlphaZero-scale reinforcement-learning strength result.
- The small-data model trained from scratch scored 7 wins, 0 draws, and 25 losses against the teacher. The reveal-variance search modification scored 14 wins, 2 draws, and 16 losses against the original engine. Neither was promoted.
- The initial warm-start candidate scored 18 wins, 1 draw, and 13 losses in a 32-game development match. Independent confirmation A produced **95 wins, 10 draws, and 95 losses (50% score)**, including one time forfeit, failing the technical-validity requirement. That confirmation set did not support the development-set advantage; confirmation B has finished at 77 wins, 17 draws, and 106 losses (42.75%), also with one time forfeit. Neither confirmation supports promotion.
- The original C++ Mist engine is now connected under its original rule profile, preserving complete player-private histories. The subsequent 16-game development screen produced **15 wins, 0 draws, and 1 loss (93.75%)**, with no void games. With only eight layout pairs and an integration timing tolerance, it does not support a formal superiority claim. Results are reported separately from matches against its JS engine. The subsequent first actual-clock 200-game set scored 192 wins and 8 losses, but four opponent time forfeits fail the predefined technical-integrity requirement. Set B completed 191 wins and 9 losses, including three opponent time forfeits. Neither set supports a formal superiority claim.

- The pure ISMCTS configuration of Yuan-Bai/jieqi-lab separately completed two 200-game sets, both 200 wins without technical failures; all 11,624 moves passed replay. Its neural model configuration remains unverified, so the whole project is not marked defeated. Enhanced Pika has ambiguous rounded-clock margins, and the Hefeng hybrid has engine-exit voids; neither confirmation is accepted.

- All [four AB-JChess comparison arms](docs/ABJ_CONFIRMATION.md) completed. The unchanged teacher scored 75.5% and 79.5% on the same A/B layouts. The learner-minus-teacher difference is +1.625 percentage points overall, with exploratory paired 95% interval [-3.125, +6.25], including zero: **a learned gain is not established**. Development and confirmation games are not pooled.

Compressed raw games, checksums, and reports are in [results](results). A [new discovery audit](docs/DISCOVERY_AUDIT.md) adds previously omitted sources. The [opponent inventory](docs/opponents.json) also retains untested, unavailable, and rule-incompatible projects. Results from different rules, time controls, or engine versions are not pooled.

The [Musesfish actual-clock compatibility profile](docs/MUSESFISH_CLOCK.md) completed a two-game integration check and 32-game development screen, all candidate wins, with all 1,397 moves/clocks replayed cleanly. Two-set confirmation matches remain pending; this establishes no strength advantage.

## Hypotheses under investigation

**Does capacity limit value learning?** We expand the same teacher function to approximately 1.64 million and 13.45 million parameters by increasing network width and adding own-king-square feature buckets, respectively. At initialization, the maximum output difference across 256 paired inputs was 9.54×10⁻⁷; moves, scores, and node counts matched on six fixed-depth native benchmarks. All three sizes use the same complete training dataset, fixed validation dataset, and learning recipe. Equal actual-clock matches then test whether improvements outweigh inference cost. Equivalent initialization does not imply greater strength after training. The first 1.64M-versus-small screen finished at 27 wins, 1 draw, and 36 losses across 64 games (42.97% score; paired bootstrap 95% interval [32.81%, 53.13%]), with no technical errors. It does not establish a capacity benefit; the 13.45M model scored 26 wins, 2 draws, and 36 losses against the same small control, also without evidence of a benefit.

**Can a larger residual network overcome a plateau?** The [capacity protocol](docs/MODEL_SCALING.md) adds wider/deeper GPU-trained models as controlled follow-up experiments. The current 9.34M model has 7.35M policy-head parameters; total size alone obscures trunk capacity. Compare validation quality, inference cost and paired equal-clock strength. A fixed 64/128-channel GPU adaptation comparison has started, with equivalent initial functions and matched data/updates; its training and subsequent paired strength screens remain incomplete. No capacity benefit is established.

**Should uncertainty determine chance-branch budgets?** We tested a variant allocating additional deep search according to pᵢ(vᵢ−E[v])², without evidence of improvement over the original engine. Further work should distinguish variance across identity outcomes, value-estimation error, and decision uncertainty that can actually change the root action, rather than allocating more computation simply because reveal outcomes differ substantially.

**How should training weight different information phases?** [Four replay-loss objectives](docs/REPLAY_WEIGHTING.md) have completed GPU training, each with the same 8,192-game source, 405,959 retained positions and 1,485 updates: equal position, equal game, equal global phase, and equal present phase within each game. The uniform control exports the exact same network bytes as the earlier experiment. The three fixed 64-game development matches scored 38.28%, 49.22% and 46.09%; none established a strength benefit.

**Can a policy head and phase-specific value experts help?** A new [residual policy/value comparison](docs/NEURAL_POLICY_PHASE.md) uses 15 residual blocks and 64 channels: a 9.34M-parameter shared-value model and a 10.07M-parameter model that switches value MLPs only when the public on-board dark count reaches zero. Both completed 12 GPU pretraining epochs on about 2.42M teacher-search positions. Policy labels are actual selected moves, not invented visit distributions. This stage is supervised warm start. All three fixed 64-game actual-clock screens completed: both models scored 0W/2D/62L against the teacher; phase versus shared scored 32W/10D/22L (57.81%; paired 95% [46.09%,69.53%]), without establishing a benefit. Subsequent self-play RL is reported separately; selection-validation error is not strength evidence.

**Do inventory and the current board omit important information?** Future experiments will compare inventory encoding with history/private-information encoding and test policy-conditioned belief updates. Actions must remain consistent within an information set, preventing decisions from using identities unavailable to the actor in sampled worlds. This remains a research direction without completed strength results.

**Can temporal targets improve learning?** A completed [same-observer two-ply target ablation](docs/TEMPORAL_TARGETS.md) partially replaces high-variance terminal labels with the same player's future search value. Data, initialization, and the validation objective are controlled across arms. The method does not directly mix values computed with the opponent's private observations. Its first 64-game development screen scored 34 wins, 2 draws, and 28 losses (54.69%; paired 95% interval [42.19%, 66.41%], still including 50%). The first direct-teacher confirmation set finished at 98 wins, 11 draws, and 91 losses (51.75%; interval [45.75%, 57.75%]), without confirming an improvement. The second set scored 93 wins, 13 draws, and 94 losses (49.75%; interval [43.5%, 56.0%]), also without supporting promotion.

**Can the transition to full observability be exploited?** The [information-phase study](docs/PARTIAL_TO_FULL.md) measures joint without-replacement identity entropy, distinguishing public certainty from actor-private certainty. A native NNUE/training ablation now ignores irrelevant off-board inventory only after all on-board dark pieces disappear. Semantic and incremental-update checks passed. The inference-only 64-game screen finished at 29 wins, 6 draws, and 29 losses, without a strength benefit; matched retraining scored 30 wins, 7 draws, and 27 losses (52.34%; paired 95% interval [42.19%, 62.5%]), also without confirming a benefit.

## Literature and methodological boundaries

- [Silver et al., AlphaZero, Science 2018](https://doi.org/10.1126/science.aar6404): motivates the cycle of self-play, search improvement, and learning. Jieqi's private information creates problems for directly applying perfect-information state search.
- [Silver & Veness, POMCP, NeurIPS 2010](https://papers.nips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf): motivates history trees, root-belief sampling, and simulator-based planning. Assumptions must be examined separately for a fixed opponent and two-player self-play.
- [Brown et al., ReBeL, NeurIPS 2020](https://arxiv.org/abs/2007.13544): studies public beliefs with search and learning, motivating explicit treatment of both players' private information and policy dependence. Its convergence guarantees do not automatically apply to this project's approximate search and function approximation.

When learning stalls, we will first investigate data coverage, target bias, belief representation, and search cost, using these papers to formulate falsifiable ablations. More training time or a larger model is not itself considered an algorithmic innovation.

## Evaluation standards

Final report gate: establish stable win rates above **70%** (wins/all games,
reported separately from draw-adjusted score) against all current verified
open-source/research strong baselines, on independent color-swapped test sets
and across training seeds. Then write an arXiv-style technical report with
training curves, core innovations, ablations and meaningful implementation
details. Current version-specific confirmations do not meet this goal; see the
[full acceptance requirements](docs/GOAL.md).

Models, versions, layout seeds, rules, clocks, threads, and sample sizes are fixed in advance; training and evaluation layouts are separated. Formal superiority requires at least 100 color-swapped layout pairs, reporting W/D/L, score rate, a 95% bootstrap interval resampled by layout pair, and all timeouts, illegal moves, voids, and truncated games. Internal promotion additionally requires a conservative bounded-mean lower confidence limit above 50% and no technical errors. Broader superiority also requires confirmation on independent layouts and validation across multiple training seeds.

Equal actual-time and equal-node budgets are studied separately. A larger network's advantage at equal nodes does not establish greater strength at equal time. Opponents must not be weakened, their official networks replaced, or their JS versions used to represent their C++ engines. Rule differences are handled through an independent referee and separate profiles; see the [native Mist profile](docs/MIST_NATIVE.md).

## Reproduction and attribution

[Continual-learning recipe and recovery](docs/CONTINUAL_LEARNING.md) · [Search ablations](docs/ADAPTIVE_SEARCH.md) · [Provenance and licenses](docs/PROVENANCE.md) · [Research goal and acceptance criteria](docs/GOAL.md)

Public records focus on algorithms, budgets, seeds, versions, and artifact checksums. Machine identifiers, usernames, absolute environment paths, and local operational details are excluded from research descriptions. Experimental weights remain candidates and should not be released as models with confirmed superiority over upstream.
