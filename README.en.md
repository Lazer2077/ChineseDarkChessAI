# Chinese Dark Chess AI

[简体中文](README.md) | **English**

Research on imperfect-information search and self-play learning for **full-board Jieqi**. We aim to outperform publicly available, runnable, independent Jieqi engines in reproducible experiments with compatible rules. The project is licensed under GPL-3.0; external engines, models, and rules are attributed separately.

**There is currently no statistical evidence of superiority across the reference engines, and no opponent has been formally marked as defeated.** The project focuses on algorithms and experimental evidence. Half-board dark chess (Banqi) is outside its scope.

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

The current series uses λ=0.7 and K=430. This is value-based expert iteration combining search-target distillation with terminal-outcome learning. It does not yet implement AlphaZero's joint policy/value learning with MCTS, or full POMCP, CFR, or ReBeL. The early 15-dimensional linear-policy prototype checks observation isolation and the search/training pipeline; it does not represent the strength of the current main experiment.

## Experimental scale and observed results

The following is an evidence snapshot dated 2026-10-26. Running jobs are not completed experiments.

- The expanded series plans three independent training seeds, two iterations per seed, and 8,192 games per iteration: 49,152 games in total. The first iteration of the first seed has completed **8,192 games and 607,652 raw position records, with zero rejected games**; filtering leaves **405,959** training positions and 25,031 independent validation positions. Its small model has completed 15 epochs and 1,485 updates, its 200-game acceptance match finished at 87 wins, 13 draws, and 100 losses (46.75%; paired bootstrap 95% interval [40.0%, 53.5%]), with no technical errors. It was not promoted, and the next generation retains the teacher. This does not establish completion of the entire series.
- Training actually uses a GPU; self-play search runs mainly on CPUs. This is currently single-GPU learning for a small NNUE, with a separate capacity ablation job. There is no evidence of distributed multi-GPU or AlphaZero-scale training.
- The small-data model trained from scratch scored 7 wins, 0 draws, and 25 losses against the teacher. The reveal-variance search modification scored 14 wins, 2 draws, and 16 losses against the original engine. Neither was promoted.
- The initial warm-start candidate scored 18 wins, 1 draw, and 13 losses in a 32-game development match. Independent confirmation A produced **95 wins, 10 draws, and 95 losses (50% score)**, including one time forfeit, failing the technical-validity requirement. That confirmation set did not support the development-set advantage; confirmation B has finished at 77 wins, 17 draws, and 106 losses (42.75%), also with one time forfeit. Neither confirmation supports promotion.
- The original C++ Mist engine is now connected under its original rule profile, preserving complete player-private histories. The subsequent 16-game development screen produced **15 wins, 0 draws, and 1 loss (93.75%)**, with no void games. With only eight layout pairs and an integration timing tolerance, it does not support a formal superiority claim. Results are reported separately from matches against its JS engine.

- Against released AB-JChess v0.2d with its official network, the expanded model scored **27 wins, 1 draw, and 4 losses (85.94%)** in 32 development games without technical errors. This is not pooled with the earlier four-game smoke under a different configuration. [Independent confirmation and unchanged-teacher controls](docs/ABJ_CONFIRMATION.md) are running to distinguish baseline strength from the contribution of training.

Compressed raw games, checksums, and reports are in [results](results). A [new discovery audit](docs/DISCOVERY_AUDIT.md) adds previously omitted sources. The [opponent inventory](docs/opponents.json) also retains untested, unavailable, and rule-incompatible projects. Results from different rules, time controls, or engine versions are not pooled.

## Hypotheses under investigation

**Does capacity limit value learning?** We expand the same teacher function to approximately 1.64 million and 13.45 million parameters by increasing network width and adding own-king-square feature buckets, respectively. At initialization, the maximum output difference across 256 paired inputs was 9.54×10⁻⁷; moves, scores, and node counts matched on six fixed-depth native benchmarks. All three sizes use the same complete training dataset, fixed validation dataset, and learning recipe. Equal actual-clock matches then test whether improvements outweigh inference cost. Equivalent initialization does not imply greater strength after training. The first 1.64M-versus-small screen finished at 27 wins, 1 draw, and 36 losses across 64 games (42.97% score; paired bootstrap 95% interval [32.81%, 53.13%]), with no technical errors. It does not establish a capacity benefit; the 13.45M model scored 26 wins, 2 draws, and 36 losses against the same small control, also without evidence of a benefit.

**Should uncertainty determine chance-branch budgets?** We tested a variant allocating additional deep search according to pᵢ(vᵢ−E[v])², without evidence of improvement over the original engine. Further work should distinguish variance across identity outcomes, value-estimation error, and decision uncertainty that can actually change the root action, rather than allocating more computation simply because reveal outcomes differ substantially.

**Do inventory and the current board omit important information?** Future experiments will compare inventory encoding with history/private-information encoding and test policy-conditioned belief updates. Actions must remain consistent within an information set, preventing decisions from using identities unavailable to the actor in sampled worlds. This remains a research direction without completed strength results.

**Can temporal targets improve learning?** A [same-observer two-ply target ablation](docs/TEMPORAL_TARGETS.md) is running, partially replacing high-variance terminal labels with the same player's future search value. Data, initialization, and the validation objective are controlled across arms. The method does not directly mix values computed with the opponent's private observations. Its first 64-game development screen scored 34 wins, 2 draws, and 28 losses (54.69%; paired 95% interval [42.19%, 66.41%], still including 50%). Independent comparisons with the original teacher are frozen and running; no improvement is confirmed.

## Literature and methodological boundaries

- [Silver et al., AlphaZero, Science 2018](https://doi.org/10.1126/science.aar6404): motivates the cycle of self-play, search improvement, and learning. Jieqi's private information creates problems for directly applying perfect-information state search.
- [Silver & Veness, POMCP, NeurIPS 2010](https://papers.nips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf): motivates history trees, root-belief sampling, and simulator-based planning. Assumptions must be examined separately for a fixed opponent and two-player self-play.
- [Brown et al., ReBeL, NeurIPS 2020](https://arxiv.org/abs/2007.13544): studies public beliefs with search and learning, motivating explicit treatment of both players' private information and policy dependence. Its convergence guarantees do not automatically apply to this project's approximate search and function approximation.

When learning stalls, we will first investigate data coverage, target bias, belief representation, and search cost, using these papers to formulate falsifiable ablations. More training time or a larger model is not itself considered an algorithmic innovation.

## Evaluation standards

Models, versions, layout seeds, rules, clocks, threads, and sample sizes are fixed in advance; training and evaluation layouts are separated. Formal superiority requires at least 100 color-swapped layout pairs, reporting W/D/L, score rate, a 95% bootstrap interval resampled by layout pair, and all timeouts, illegal moves, voids, and truncated games. Internal promotion additionally requires a conservative bounded-mean lower confidence limit above 50% and no technical errors. Broader superiority also requires confirmation on independent layouts and validation across multiple training seeds.

Equal actual-time and equal-node budgets are studied separately. A larger network's advantage at equal nodes does not establish greater strength at equal time. Opponents must not be weakened, their official networks replaced, or their JS versions used to represent their C++ engines. Rule differences are handled through an independent referee and separate profiles; see the [native Mist profile](docs/MIST_NATIVE.md).

## Reproduction and attribution

[Continual-learning recipe and recovery](docs/CONTINUAL_LEARNING.md) · [Search ablations](docs/ADAPTIVE_SEARCH.md) · [Provenance and licenses](docs/PROVENANCE.md) · [Research goal and acceptance criteria](docs/GOAL.md)

Public records focus on algorithms, budgets, seeds, versions, and artifact checksums. Machine identifiers, usernames, absolute environment paths, and local operational details are excluded from research descriptions. Experimental weights remain candidates and should not be released as models with confirmed superiority over upstream.
