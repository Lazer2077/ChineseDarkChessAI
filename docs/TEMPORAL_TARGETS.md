# Same-observer temporal targets: a frozen ablation

The first expanded learner scored 87W/13D/100L against its teacher in 200 games,
with no technical failures and a paired bootstrap 95% score interval of
[0.400, 0.535]. It was not accepted; the next generation retains the teacher.
Same-data 1.64M and 13.45M capacity screens scored 27W/1D/36L and 26W/2D/36L
against the trained small control. Neither supports a capacity benefit.

## Diagnosis before changing the objective

On the 25,031-position checkpoint-selection validation set, the original
teacher's outcome Brier loss is 0.118222; the selected small and large models
reach 0.117874 and 0.117812. For the teacher, loss is 0.043632 with no hidden
on-board pieces, versus 0.196893 with 15–30 hidden pieces. These are descriptive
stage strata, not causal evidence that hidden identities alone cause the gap;
game phase and position difficulty are confounders. Positions are correlated
within games, and this validation set is not an independent strength test.

The small numerical validation improvement alongside absent clock-match gains
motivates testing target construction, rather than assuming more capacity is
enough. Raw diagnostics: results/20261026-value-diagnostics.json.

## Literature and observer choice

[TDLeaf](https://arxiv.org/abs/cs/9901001) combines temporal-difference learning
with game-tree search. [Expert Iteration](https://arxiv.org/abs/1705.08439)
separates planning from generalization. These motivate a temporal search-target
ablation; our implementation does not differentiate through search leaves or
maintain eligibility traces and is not a full TDLeaf implementation.

A tempting alternating-player negamax bootstrap would use a successor score
computed with the opponent's private inventory. It is not the same observer's
POMDP value. We instead use the next recorded decision of the **same player**,
two plies later, including only that player's then-available private knowledge.
Future observations may inform a learning target; they are not supplied as
inputs when acting at the current observation. This is an observation-space
TD-style target on fixed-teacher trajectories, without Bayesian optimality or
Nash-equilibrium guarantees. The target still depends on the behavior policy
and an approximate teacher; reduced variance or improved strength is a
hypothesis, not an established result.

## Frozen target and controls

Let q_t=σ(s_t/430) be the teacher's search value from the acting player's own
observation, and y=(z+1)/2 the realized final score. The control target is
0.7 q_t + 0.3 y. The temporal variant uses:

```math
y_t^{\mathrm{temporal}}=0.7q_t+0.15y+0.15q_{t+2}^{\mathrm{same\ observer}}.
```

If no later decision by that player is recorded before the terminal result,
the final-score target replaces q_{t+2}. Opponent-observer scores are never
negated into the target. Terminal draws remain 0.5.

Both variants use the same 8,192 completed games, 405,959 retained current
positions, teacher initialization, architecture, learner seed, 15 epochs,
batch 4096 and learning rate 2e-5. Both select checkpoints using **the identical
control-target validation file**, so changing the training objective does not
also change the selection objective. The paired development match is fixed
at 64 games, layout seed 7801001, actual clock 5+0.05, one thread/64 MB each.
No promotion is claimed from this single-seed screen.

The original binary record's score field stores a synthetic target logit in
the derived files; it is not an actual search score. Both derived variants
therefore use the unchanged trainer with --lam=1 and K=430. Encoding introduces
maximum probability error 0.000291, explicitly measured for both variants.
The control is also relabeled, so both arms share this encoding mechanism.
Keep derived files separate from original search records.

## Trajectory certification

The actor writes each completed game's consecutive-ply buffer atomically, but
the old binary format lacks explicit game IDs. Ply resets alone found only
8,191 boundaries for 8,192 games: one cross-game boundary happened to have
consecutive ply numbers. Training on that reconstruction was refused.

The extractor now also checks the public-board transition implied by the
recorded move: only source/destination may change, the source becomes empty,
and the destination must contain the original piece or its legal-color public
reveal. Combining these checks with ply continuity certifies exactly 8,192
training games and 512 validation games against complete-generation counts
and data hashes. It verifies consistent final labels and side/ply parity.
Any unmatched count is rejected; boundaries are never guessed to force a run.
403,722 retained training rows have a later same-observer search target.

Tests cover a consecutive-ply cross-game boundary, public reveal transitions,
terminal fallback, inconsistent labels, and invariance of red targets to
changes in black-observer search scores. The next data format should store
explicit game IDs and terminal reasons to avoid reconstruction entirely.

Frozen recipes and artifact hashes: results/temporal-target-plan.json. Local
training outputs are under runs/temporal-ablation-01; queued or running jobs
are not strength evidence.
