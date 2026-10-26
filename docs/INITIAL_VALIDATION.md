# Initial validation — 2026-10-26

Runtime: Node 22.23.4; PyTorch 2.13.0+cu130 from an existing environment.
CUDA allocation and tensor creation verified on an RTX A5000.

- Three research tests pass: hidden-world invariance; captor-only private capture
  visibility and complete inventory sampling; legal seeded selfplay and policy schema.
- Two initial deterministic smoke selfplay games reached terminal repetition outcomes.
  This exposed insufficient exploration at argmax selection; selfplay now samples
  from visit distributions. Evaluation keeps argmax selection.
- Three CPU training epochs over 32 smoke rows produced a reloadable policy;
  training objective decreased from 4.7046 to 4.2983. This only checks optimization.
- Reloaded model against the pinned upstream JS baseline: one candidate loss and
  one 40-ply truncated game. Unequal budgets, tiny sample: no strength conclusion.
- First experiment launched in a persistent tmux session `jieqi-research`:
  8 gen0 games -> 20 GPU epochs -> 8 gen1 games -> 20 GPU epochs ->
  8 games against upstream JS at 100 ms/move. Search uses 128 simulations
  (at least one visit per legal root action), rollout horizon 4, max 400 plies.
  Training seed/layout bases: 20261026, 20262026; evaluation: 30361026.
  This is a pipeline pilot with unequal compute, not a promotion benchmark.

Raw outputs/checkpoints/logs remain local under `runs/first-experiment/`.
No claim of completed training, convergence, or improvement is made here.
The policy is linear; value head is diagnostic and not used by search.
