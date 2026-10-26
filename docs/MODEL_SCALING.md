# Residual model capacity after a strength plateau

The user explicitly added model scaling as an available response to a win-rate
plateau on 2026-10-26. This extends the active research goal; it does not establish
that the current bottleneck is capacity. Complete frozen experiments unchanged.
The following is a candidate protocol, not a launched or completed training run.
Freeze concrete data, seeds, initialization, budgets and comparisons before launch.

## Architecture audit

Direct parameter enumeration of the pinned Yuan residual architecture through
`research/policy_value.py`, with a shared value head, gives:

| Channels | Residual blocks | Total parameters | Residual trunk parameters |
| --- | --- | --- | --- |
| 64 | 15 | 9,335,543 | 1,109,760 |
| 96 | 15 | 10,790,519 | 2,494,080 |
| 128 | 15 | 12,798,455 | 4,431,360 |
| 64 | 30 | 10,445,303 | 2,219,520 |

The current model's policy head contains 7,348,662 parameters, value head 739,905
and input stem 137,216. The flattened 2,880-to-2,550 policy linear layer dominates
total size. Doubling trunk width therefore quadruples residual-block parameters
without quadrupling total parameters. Parameter counts are not measured FLOPs,
latency or GPU-memory requirements. These alternatives are architecture counts,
not trained checkpoints.

## Controlled follow-up

- Start with the shared-head model. Vary width or depth independently, keeping
  observation contract, legal actions, search rules and public-phase handling fixed.
  Enlarging a network cannot recover history omitted from its input.
- Use identical game-level training/selection splits and augmentation. Record
  train and validation policy loss, terminal-value error and phase-stratified
  errors. Selection data cannot become a fresh strength test.
- Separate matched-data/update comparisons from matched training-compute
  comparisons. If using function-preserving expansion, verify initial policy and
  value agreement and compare a small model continued from the same checkpoint.
  Otherwise distinguish random-initialization training from warm-start effects.
- Train on GPUs and record actual optimization steps, games, elapsed training
  time, memory, batch size and numerical precision. Replicate promising results
  across training seeds before attributing gains to capacity.
- Measure inference latency/throughput at batch 1 and search leaf-batch sizes,
  plus completed search simulations under actual game clocks. Fixed-simulation
  diagnostics isolate search quality; paired equal-clock games establish whether
  better predictions compensate for slower inference. Disclose GPU/CPU resources.
- Re-enter on-policy self-play only with explicit actor/checkpoint provenance.
  Use fresh paired layouts against the small control and unchanged teacher;
  apply the existing confirmation gate before promotion and the separate >70%
  strong-baseline gate before the requested technical report.

Earlier widened/bucketed NNUE controls did not establish a gain. They do not
settle capacity scaling for this different residual policy/value architecture.
Current neural self-play already uses GPU search and GPU optimization; adding
parameters and increasing self-play data or search effort remain distinct
experimental interventions. No scaling-law or strength claim follows from the
architecture audit.
