# Game and information-phase weighting ablation

Current self-play value learning averages retained positions. Longer games and
longer phases therefore contribute more total loss. The phase diagnostics show
that game-weighted and position-weighted errors can differ, motivating a change
in the training objective rather than assuming that more parameters are enough.
This is a standard stratified weighting experiment, not a new RL convergence
theorem or an implementation of equilibrium learning.

The four frozen objectives are:

```math
L_{\rm position}=\frac1N\sum_i\ell_i,\qquad
L_{\rm game}=\frac1G\sum_g\frac1{n_g}\sum_{i\in g}\ell_i,
```

```math
L_{\rm phase}=\frac1P\sum_p\frac1{n_p}\sum_{i\in p}\ell_i,\qquad
L_{\rm game,phase}=\frac1G\sum_g\frac1{|P_g|}
  \sum_{p\in P_g}\frac1{n_{g,p}}\sum_{i\in(g,p)}\ell_i.
```

Here G counts games with retained rows; P and P_g contain only nonempty phases.
The loss is the existing squared probability error against 70% approximate search
value plus 30% terminal outcome. Four phases use the **public** on-board dark
count: 0, 1–4, 5–14, and 15–30. This is an inexpensive observable proxy, not
exact identity entropy or proof that zero entropy is common knowledge.

`scripts/build_replay_weights.py` verifies complete trajectories, hashes and
unchanged upstream filtering. The 8,192-game source leaves 405,959 positions
from 8,187 games: five games have no rows after filtering and cannot contribute
to any arm. Weight files align with the same retained row order. We multiply
each squared error by a fixed positive weight whose **global** mean is one,
then take the batch mean. There is no per-batch normalization or replacement
sampling. The uniform control uses all-one weights through the same trainer.

| Objective | Full-public loss mass | Largest record weight | Weight ESS |
| --- | ---: | ---: | ---: |
| Position | 10.19% | 1.00 | 405,959 |
| Game | 5.05% | 49.59 | 266,748 |
| Global phase | 25.00% | 2.45 | 321,067 |
| Game, then present phase | 4.33% | 49.59 | 205,886 |

Equalizing phases **within games** does not equalize global phase exposure:
fully public positions are concentrated in relatively few, longer games. This
is why the global-phase arm is separate. Weight ESS is the algebraic quantity
(sum w)^2/sum(w^2), not the number of independent samples or games. Large weights
in the game arms can increase gradient variance; no retrospectively chosen
clipping threshold is introduced in this fixed experiment.

## Controlled GPU experiment

All four arms start from the same attributed jqe teacher and use identical data,
unweighted validation, random seed, shuffled rows, mirror augmentation, small
NNUE architecture, optimizer, schedule, learning rate and export path. Each arm
runs 15 epochs / 1,485 updates with batches of 4,096, learning rate 2e-5 and
score scale 430. Checkpoints minimize the common unweighted validation loss.
Whole-epoch shuffling drops the same final incomplete batch in every arm, as
in the original trainer. No labels, legal moves or game histories change.

`scripts/build_weighted_trainer.py` creates a local MIT-attributed copy of the
pinned trainer, modifying only the optional training loss weights; the upstream
checkout and all opponents stay unchanged. Unit checks verify equal game mass,
equal phase mass, the distinction between global and within-game phase balancing,
and the exact weighted objective on a hand-computed example.

After training all four arms, three fixed 64-game development comparisons use
the same new color-swapped layout pool against the uniform control, actual
clock 5+0.05, one thread and 64 MB hash. All arms must finish. These small,
multiple-comparison screens cannot promote a candidate; independent larger
teacher and other-opponent confirmation is still required. Frozen recipe:
results/replay-weight-plan.json. A planned arm is not a completed result.

## Relation to replay literature

[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) motivates
examining which replay transitions drive learning. Our priorities are fixed by
game/observable phase, not TD error. We intentionally change the empirical
objective instead of claiming importance-corrected estimation of the original
uniform-position objective. This does not correct off-policy behavior, imperfect
teacher labels or the gap between a value-regression loss and stronger play.
The game-phase interpretation is specific to the gradually revealed structure
being investigated here; its benefit remains an experimental hypothesis.

## Completed training, playing comparisons pending

All four arms completed 15 GPU epochs and 1,485 updates. Selected epochs are
1, 2, 1, 2 for position, game, global phase and game-phase, respectively. Common
unweighted validation losses are 0.01410217, 0.01411622, 0.01410315 and 0.01413381.
The uniform control net SHA256 is exactly the prior expanded-small net hash
9cd00a5f4fb7ccceb5c1374660a623acb60cd18eafa79f4c6aa9848b0e278833.
This verifies the control path, not strength. Full-public phase diagnostics
remain exploratory on checkpoint-selection data. Evidence:
results/20261026-replay-weight-training.json and
results/20261026-replay-weight-diagnostics.json. Fixed playing comparisons continue.

The first game-weighted versus position-weighted development match has now
completed: 24W/1D/39L over 64 games / 32 pairs, score 38.28125%, paired bootstrap
95% [0.265625, 0.5078125], with no technical failures or voids. It does not
support a benefit from equal-game loss weighting. The global-phase and
within-game phase arms continue unchanged; no retuning follows this outcome.
Evidence: results/20261026-replay-game-development.json.

Both remaining fixed development arms completed without technical failures or
voids. Global-phase weighting scored 30W/3D/31L (49.21875%; paired bootstrap
95% [0.375, 0.609375]); within-game phase weighting scored 28W/3D/33L
(46.09375%; [0.359375, 0.5625]). All three weighted arms failed to establish
a benefit over the position control. No weighted arm is promoted. Evidence:
results/20261026-replay-phase-development.json and
results/20261026-replay-game-phase-development.json. The fully public transition
remains a research property; this particular reweighting does not yet exploit it
with demonstrated strength gains.
