# Chinese Dark Chess AI

[简体中文](README.md) | **English**

Research on partial-observation search, policy/value learning and self-play in **full-board Jieqi**. The goal is reproducible superiority over publicly runnable independent Jieqi engines under compatible rules. Half-board Banqi is outside scope.

**The confirmed learning gain is against an internal neural reference; superiority over the original strong teacher remains unproven.** The new pretrained model records **158 wins, 6 draws and 36 losses** in a fresh 200-game confirmation against the old neural model. Its development result against the original jqe teacher remains **0 wins, 2 draws and 62 losses**. Both results matter.

## Problem and information boundary

Jieqi is a two-player zero-sum partially observable stochastic game. With a fixed opponent policy and appropriate state augmentation, one player's planning problem can be viewed as a POMDP; this does not directly give an equilibrium guarantee for changing self-play policies.

A player observes the public board, actions, reveals and identities of dark pieces that player captures. Identities captured by the opponent may remain unknown. Hidden types are allocated without replacement and are correlated. Opponent actions can also reveal information about private observations; the current inventory approximation does not fully solve this policy-conditioned inference problem.

The physical count of live dark pieces decreases during play. Once both players have no live dark pieces, moves and terminal rules no longer depend on unknown off-board identities. We clear irrelevant inventory inputs at that exact boundary while preserving complete referee history. Entropy over unknown captured inventories and uncertainty about the opponent's private information need not decrease monotonically.

## Current neural research

A residual policy/value network guides actor-information PUCT: sample hidden worlds from the legal observation, share decision statistics by the simulated actor's information, and represent reveal chance branches separately. Search, rules and the initial architecture derive from pinned [Yuan-Bai/jieqi-lab](https://github.com/Yuan-Bai/jieqi-lab). Teacher search and published it014 weights come from [devinokeefe/jieqi](https://github.com/devinokeefe/jieqi). See [provenance](docs/PROVENANCE.md) for attribution and licensing boundaries.

Network inputs contain the current public board, actor-known inventories and counters. Historical features absent from old replay remain empty in both training and inference; search still retains complete real history. This encoder is not a sufficient belief state, and the implementation does not claim full POMCP, CFR or ReBeL.

### Spatial policy decoder and wider shared backbone

The [relative-action decoder study](docs/SPATIAL_POLICY_HEAD.md) follows the established [Lc0 convolution-and-gather principle](https://lczero.org/dev/backend/nn/), mapping the existing 2,550 geometric actions to 50 displacement planes. Actual movement rules stay intact, including fake movement geometry before revelation. Sharing weights does not assume that arbitrary translations preserve palaces, rivers or legal game symmetries.

| Model | Shared backbone | Total parameters | Final policy classifier |
| --- | --- | ---: | ---: |
| Old neural reference | 64 channels × 15 blocks, dense decoder | 9,335,543 | 7,346,550 |
| Matched dense control | 128 channels × 15 blocks | 12,798,455 | 7,346,550 |
| New relative model | 128 channels × 15 blocks | 5,453,555 | 1,650 |
| Capacity study, training | 256 channels × 20 blocks | 24,918,259 | 1,650 |

The two 128-channel arms share identical common initialization, data order and optimization recipe; their policy classifiers differ. Relative to the old 64-channel model, backbone width, initialization and fitting also change, so the full playing-strength difference cannot be attributed to the decoder alone.

### Supervised pretraining and self-play RL are separate

Pretraining uses **2,417,541 actual teacher-search positions** from 32,768 generated games, of which 32,767 contain searched records. No labels are invented for unsearched random opening moves. Policy supervision is the actual selected teacher action; value supervision mixes the actor-relative search score and terminal return:

```math
y=0.7\tanh(s/860)+0.3z,\qquad
\mathcal L_{pre}=\mathrm{CE}(a_{teacher},p_\theta(o))+(v_\theta(o)-y)^2.
```

Each new model uses one training seed, 12 epochs, batch 512 and 56,652 updates, with the same optimizer, augmentation and learning-rate schedule. The common selection CE+MSE rule chooses epochs 8/10 for the two 128-channel arms. Selection-set teacher-action top1 is 44.85% for relative and 39.20% for dense. These checkpoint-selection metrics are neither independent generalization nor playing-strength evidence.

Current-model self-play is a separate stage, fitting actual completed root-visit distributions and terminal returns:

```math
\mathcal L_{RL}=\mathrm{CE}(\pi_{search},p_\theta(o))+(v_\theta(o)-z)^2.
```

This is AlphaZero-style search-policy/return fitting, not a claim of unbiased asymmetric policy gradients. Full curves, data roles and frozen artifact digests are retained in [results](results).

## Completed neural matches

Evidence as of 2026-10-27. Use actual **10+0.1** clocks, neural leaf batch 32, one CPU thread per engine and two concurrent games sharing an inference device. Intervals resample complete color-swapped layout pairs; score means `(wins+0.5 draws)/games`.

| Frozen comparison | Stage | W/D/L | Score | Paired score 95% interval |
| --- | --- | --- | ---: | --- |
| New relative versus old neural reference | Development, 64 games | 52/1/11 | 82.03% | [71.88%, 90.63%] |
| Same frozen model versus old neural reference | Fresh confirmation, 200 games | 158/6/36 | 80.50% | [75.25%, 85.50%] |
| Relative versus matched dense 128 | Development, 64 games | 38/1/25 | 60.16% | [48.44%, 71.88%] |
| Relative versus original jqe teacher | Development, 64 games | 0/2/62 | 1.56% | [0%, 3.91%] |
| Dense 128 versus original jqe teacher | Development, 64 games | 1/0/63 | 1.56% | [0%, 4.69%] |

The 200-game confirmation has a strict **79% win fraction**, paired win-fraction 95% interval **[73.5%, 84.5%]**. Development and confirmation layout seeds are disjoint. All 12,631 moves across those two campaigns pass identity, legality, terminal and clock replay, with zero technical games or clock-rounding ambiguities. The sets remain separate rather than being pooled into one confirmation.

The supported conclusion is superiority of this frozen model over the old internal neural reference. The matched dense interval still includes 50%, so a decoder-only playing-strength effect is unconfirmed. There is no strong-teacher or final-suite victory. See [complete evidence and raw-record digests](results/20261027-spatial-relative-confirmed-reference-gain.json).

## Active controls and negative evidence

- **Why does self-play regress?** One ordinary 288-game/448-update iteration of the relative model records 20/2/42 against its pretrained parent and 0/0/64 against the teacher. Lower selection losses do not imply stronger play. [Study](docs/SPATIAL_SELFPLAY.md)
- **Are rules and terminal targets aligned?** Public-history replay finds earlier native terminals in 19 of 256 training games and different terminal returns in 12. [Native-terminal prefix controls](docs/NATIVE_TERMINAL_REANALYSIS.md) admit only legal prefixes that genuinely reach native terminal states and compare return labels on matched observations/visits. This aligns realized trajectories, not all counterfactual search dynamics.
- **Should exploratory visits become policy targets directly?** [Policy-target pruning](docs/POLICY_TARGET_PRUNING.md) follows [KataGo §5.3](https://arxiv.org/html/1902.10565v1). A fixed 32-TRAIN-root teacher-score proxy gives a positive signal. Exact reproduction then covers all 288 games/15,331 native-prefix roots, retaining actual counts and archiving separate targets. The raw-control model reproduces bitwise; both fits complete 428 updates. The first raw-control screen completes only 63 games plus one startup void and is not a valid 64-game screen; all records are retained while the original parent/teacher contrasts continue. Proxy benefit is not strength; KataGo forced playouts are not implemented.
- **Does capacity limit progress against the teacher?** The [fixed capacity study](docs/SPATIAL_CAPACITY.md) trains 256 channels × 20 blocks on the same teacher data and 56,652 updates. Full-size GPU update, numeric interchange and search integration pass. Prediction quality must offset inference cost under actual clocks before claiming stronger play.
- **May a teacher know hidden identities?** [Training-time privileged information](docs/PRIVILEGED_TRAINING.md) is conditionally defensible, but clairvoyant action imitation can mismatch the student's information-gathering needs. A completed equal-capacity value study has 64-game held-out MSE 0.8954 symmetric versus 0.8979 privileged, with no benefit; that specific branch is not expanded. Deployed models receive no true dark identities.

Other closed or inconclusive directions retain their evidence and limits: [normalization drift](docs/NORMALIZATION_DRIFT.md), [virtual mean](docs/VIRTUAL_MEAN.md), [observer-history expert relabeling](docs/OBSERVER_EXPERT_RELABEL.md), [GPU inference execution](docs/INFERENCE_EXECUTION.md) and [public-phase search](docs/PARTIAL_TO_FULL.md). Pipeline checks, lower proxy losses and wins over regressed controls are not strong-baseline breakthroughs.

## Open-source coverage and earlier native results

The [opponent inventory](docs/opponents.json) and [discovery audit](docs/DISCOVERY_AUDIT.md) track versions, licenses, rules, weights and runnability, deduplicating shared engine cores. Unrunnable, incompatible or insufficiently licensed projects remain unverified; exclusion is not a victory.

[Opponent-only preselection](docs/BASELINE_SELECTION.md) currently retains original jqe, AB-JChess v0.2d and the native classic Pika WASM branch as routine baselines. AB and Pika share engine-family ancestry, so these are not three independent algorithm families. Newly found strong candidates still require review.

An earlier native NNUE candidate completed two 200-game confirmations per fixed version below, at actual 5+0.05:

| Fixed version | Confirmation A, W/D/L | Confirmation B, W/D/L |
| --- | --- | --- |
| AB-JChess v0.2d | 162/7/31 | 150/2/48 |
| universe-st/Jieqi `e00b667c` | 200/0/0 | 198/2/0 |
| brianhliou/Pika WASM native build `bcc83f88` | 181/1/18 | 179/3/18 |

Those results belong to that earlier native candidate, **not to the new GPU policy/value model**, and do not establish a learning gain over its original teacher. Other configurations, technical failures and clock ambiguities remain separately reported in [confirmed results](docs/CONFIRMED_RESULTS.md), [teacher attribution controls](docs/ABJ_CONFIRMATION.md) and [baseline selection](docs/BASELINE_SELECTION.md).

## Evaluation and final-report gate

Freeze versions, model, observation permissions, referee, clock, threads, layout seeds and sample size before evaluation. Retain W/D/L, strict win fraction, draw-adjusted score, paired intervals and all technical outcomes. Separate training, checkpoint selection and strength-test roles. One training seed per experiment is permitted; fresh test layouts provide independent evaluation.

The final goal requires the **same frozen model** to exceed **70% wins** against **each final strong baseline** on at least two independent test-layout sets, with each paired 95% win-fraction lower bound above 70%, at least 100 color-swapped layout pairs per set, and clean rule/identity/actual-clock audits. Only then write an arXiv-style technical report with training curves, core innovations, ablations and relevant implementation details. **This gate is not met.** See the full [persistent goal](docs/GOAL.md).

The project uses GPL-3.0; imported engines, models, data and rules retain their separate attribution. Method references include [AlphaZero](https://doi.org/10.1126/science.aar6404), [POMCP](https://papers.nips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf), [ReBeL](https://arxiv.org/abs/2007.13544) and [KataGo](https://arxiv.org/abs/1902.10565). Their guarantees do not automatically apply to the current approximate Jieqi implementation.
