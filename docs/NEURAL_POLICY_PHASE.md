# Residual policy/value pretraining with public-phase value experts

This experiment moves beyond the earlier value-only NNUE updates. It adds a
policy head and a separate value-function hypothesis for the fully public
continuation. This stage is **supervised warm start from teacher self-play**,
not completed reinforcement learning, a new equilibrium algorithm or evidence
of stronger play. The current main actor is still the attributed jqe teacher.

## Architecture and controlled comparison

The residual trunk, policy/value architecture and 2,550-action geometry come
from MIT-licensed `Yuan-Bai/jieqi-lab`, commit
`4173aa3ddc72e19227a619ffc4ea80a88b3ba2b4`. The source checkout stays unchanged.
The original MIT notice is retained in `licenses/Yuan-Bai-jieqi-lab.MIT.txt`.
Our research code imports that architecture and makes its input adaptation and
value-MLP modification explicit.

Both arms have 15 residual blocks with 64 channels and one policy head:

- Shared control: one value MLP, 9,335,543 parameters.
- Phase experts: two value MLPs after shared trunk/value-convolution features,
  10,073,336 parameters. The public MLP is copied from the ordinary value MLP,
  preserving the complete initial policy/value function.

Only the **public count of on-board dark pieces equal to zero** selects the
public MLP. Private inventory certainty does not trigger the switch. Both arms
canonicalize irrelevant off-board inventory in that phase, so the comparison
isolates value-MLP specialization under the same input abstraction. The policy
head and convolutional features remain shared. Initialization, data, augmentation,
update count, optimizer and checkpoint objective are controlled. Parameter counts
and computation differ; this is not an equal-FLOP capacity claim.

## Observations and labels

The source is four completed native teacher datasets: 32,768 generated games and
2,417,541 searched positions, with zero rejected generation games. Each record
contains the acting player's public board and private inventory, never the
simulator's unrevealed true identities. Dark home movement roles are public and
are reconstructed from their still-occupied original squares.

The input includes board roles/types, side to move, observer inventory, live dark
counts, unknown removal counts and the recorded no-capture counter. The source
records omit their random opening moves and full rule history. We consequently
omit recent-move, repetition and check/chase history features **consistently in
training and inference**, leaving the complete-history indicator false. We do
not turn missing history into a claim of known zero rule counters. This is a
compressed observation critic, not a sufficient statistic for the full game.
The search actor must retain the complete real rule/history state separately.

Policy supervision is the actual search-selected action, with a one-hot cross
entropy target. No search visit distribution is invented. Value supervision is

```math
y_v=0.7\tanh(s/860)+0.3z,\qquad
L=\operatorname{CE}(\pi,a_{teacher})+(v-y_v)^2.
```

This is the earlier probability-space target transformed to [-1,1], from the
acting player's perspective. Policy/value training includes searched captures,
checks and mate-score positions, unlike the previous static NNUE filter. Thus
its validation loss is not directly comparable to the earlier filtered NNUE
loss. The source actor uses its internal self-play adjudication, including a search-
cycle approximation, rather than replaying every generated episode through the
independent benchmark referee. Outcome labels inherit that limitation.
Approximate teacher bias remains, including the teacher's dependence on
dead-inventory features in fully public states. Canonical inputs do not make
those teacher labels exact optimal-continuation values.

The common selection-validation set contains 512 games / 37,471 searched rows,
including 4,570 fully public positions. It has been used in earlier research;
it is checkpoint-selection data, not a fresh independent strength test. Policy
agreement and value error do not establish playing strength.

## A verified zero-row episode

The first launch stopped before any optimization: one source had 8,191 record
segments against 8,192 generated games. We did not merge boundaries or drop data.
`scripts/check_empty_replay_games.py` replays the exact native actor's seeded
random opening and exits immediately before the first searched decision. It
certified one game ending before any search record, in seed20261027 iteration2,
game index1492. The other three training files and validation have no zero-row
games. Therefore 32,767 training games contribute the existing 2,417,541 rows.

The loader now verifies the certificate against source/core/controller hashes,
seed and dataset digest, then certifies exactly the expected **nonempty** game
segments and within-game labels. Every recorded row is retained. The failed
preflight and its unchanged plan remain archived; no failed attempt is reported
as GPU training. Evidence:
`results/20261026-empty-replay-certificate.json` and
`results/20261026-policy-preflight-failure.json`.

## Fixed GPU training

Both arms run on separate GPUs for 12 epochs, batch512, AdamW learning rate3e-4,
weight decay1e-4, gradient clipping1, five-percent warmup and cosine decay. The
same dedicated RNG controls row shuffling and horizontal mirroring. There are
4,721 updates per epoch / 56,652 planned per arm; initial states are not eligible
selected checkpoints. Both arms select the minimum common validation CE+MSE.
No early stop or retuning follows intermediate results. Invalid gradients abort
rather than silently count skipped optimizer steps. Epoch checkpoints and logs
are written atomically. Planned work is not completed work.

Frozen recipe: `results/neural-policy-phase-plan.json`.

Both arms have now completed all 12 epochs / 56,652 optimizer updates, without
nonfinite-gradient failure. The common selection objective chooses epoch 10 in
both arms. Completed logs and selected array hashes are archived in
`results/20261026-neural-policy-phase-training.json`.

| Selection-validation metric | Shared value | Public-phase expert |
| --- | --- | --- |
| Policy CE | 2.36509 | 2.36145 |
| Value MSE | 0.08444 | 0.08731 |
| Fully public value MSE | 0.06782 | 0.07975 |
| Teacher action top1 agreement | 35.03% | 34.78% |

The phase expert does not improve fully public value error in this comparison.
The tiny difference in the combined selection objective is not a strength gain.
These reused selection data are correlated within games; no position-wise
independent confidence claim is made. Three predeclared 64-game CPU-only
development matches are running at a common actual 10+0.1 clock: shared versus
teacher, phase versus teacher on the same layouts, and phase versus shared on
a separate layout set. All arms finish without outcome-based retuning. Recipe:
`results/neural-policy-phase-evaluation-plan.json`. Matches use one CPU thread
and charge complete public-position reconstruction/search wall time. The neural
tree ignores the classical Hash option, so equal hash memory is not claimed.

## Search integration and correctness checks

`research/neural_search.py` reuses the pinned root-information PUCT, with a
deadline checked only between fully backed-up leaf batches. Its neural evaluator
uses the same compressed input contract as training; the original search still
keeps history, legality, belief updates and information-set keys. Without a
deadline, fixed-simulation policy/root statistics match the original search with
the same evaluator. Changing actual opponent-private ledger slots leaves root
search unchanged. Expired-budget tests complete a real batch with normalized
visit targets, without inventing visits or backing up unfinished batches.

Models export named numeric NPZ arrays, loaded with `allow_pickle=False`, with
explicit shape/dtype/finite checks. The CPU-first UCI actor uses one CPU thread;
GPU search is a separate optional profile. The current integration probe uses
only a tiny pipeline-test checkpoint and is not a strength result. No new
neural model is promoted before complete-history, actual-clock paired matches
against the teacher and other opponents. The inherited search's approximate
information model and internal rule differences remain research limitations;
we do not inherit AlphaZero/ReBeL/Nash convergence guarantees from the architecture.

The CPU-only end-to-end compatibility probe completed two games using a tiny
five-update pipeline checkpoint, with39 legal moves/clock charges independently
replayed. It lost both games against the teacher, as recorded, and is not either
full training arm or a strength evaluation. Evidence:
`results/20261026-policy-value-integration-probe.json`.

## Relation to search-and-learning research

[Expert Iteration, Anthony et al. 2017](https://arxiv.org/abs/1705.08439)
separates search improvement from network generalization, then closes the loop
by guiding subsequent search with the learned network.
[AlphaZero, Silver et al. 2017](https://arxiv.org/abs/1712.01815) uses self-play
search policies and terminal values in perfect-information games. Our current
CNN stage supplies only a supervised initialization; actual search visit targets,
new-model self-play and measured policy improvement are still required to close
that loop. Root determinization and an information-keyed tree do not by
themselves solve strategic signaling or justify perfect-information guarantees.
