# Residual model capacity after a strength plateau

The user explicitly added model scaling as an available response to a win-rate
plateau on 2026-10-26. This extends the active research goal; it does not establish
that the current bottleneck is capacity. Complete frozen experiments unchanged.
The first fixed two-arm continued-pretraining experiment is running. Its recipes
and follow-up comparisons were frozen before optimizer outcomes; neither training
completion nor a playing-strength benefit is claimed.

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
  time, memory, batch size and numerical precision. One training seed is sufficient
  under the amended goal. Confirm promising checkpoints across independent test
  seeds; report capacity findings as conditional on the tested training run.
  Additional training-seed replication is optional, not a promotion prerequisite.
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

## First fixed width experiment

The completed initialization audit is
`results/20261026-residual-width-initialization.json`: 128 partial and 128 fully
public inputs from existing selection replay, CPU FP32. The maximum policy-logit
difference was 2.29e-5 and value difference 9.54e-7. Width doubling duplicates
residual channels and batch-normalization state, splitting subsequent input
weights. Opposite seeded perturbations sum to zero across each duplicated input
pair, preserving the initial function while allowing gradients to diverge.
This adapts [Net2Wider](https://arxiv.org/abs/1511.05641), not a claimed original
algorithm. Initialization tests also cover training-mode batch normalization,
source/RNG preservation, unequal gradients and safe numeric checkpoint export.

Both arms continue from the previously selected shared residual warm start,
with fresh optimizer states, identical existing 2,417,541-row teacher replay,
512-game selection data, two epochs, batch 512, 9,442 updates, AdamW peak learning
rate 1e-5, and shared augmentation seed 20261030. The wider model has 128 channels
and 15 residual blocks. This is supervised capacity adaptation, separate from
the concurrently running on-policy neural self-play series. Two epochs and one
seed can screen a method but cannot exclude benefits with longer or independent
training. The unchanged trainer uses BF16 GPU autocast here.

Recipes: `results/residual-width-{small,wide}-training-plan.json` and
`results/residual-width-series-plan.json`. Arms run sequentially on one GPU with
matched examples/updates; this is not a matched-FLOP or randomized timing study.

The fixed follow-up `results/residual-width-evaluation-plan.json` first measures
FP32 forward-only latency at batch 1/32 (10 warmups, 50 repetitions), then runs
64-game paired actual 10+0.1 development screens: wide versus small, small versus
unchanged teacher, and wide versus unchanged teacher, with disjoint seeds
8470001/8480001/8490001. Neural actors use leaf batches 32 and the same GPU;
each actor has one CPU thread, with two games concurrent. Teacher NNUE is CPU-only;
no equal-total-compute claim is made. Full native legal/identity/terminal and clock
replays follow each match. Model hashes are recorded after checkpoint selection
and before play; no replacement checkpoint is allowed after outcomes. A waiting
evaluator verifies the training controller remains live and never restarts it.

The small continuation arm has completed both epochs / 9,442 updates; epoch 1 was
selected (selection policy CE 2.3588523, value MSE 0.0842577). Its full curves and
checkpoint digest are archived in
`results/20261026-residual-width-small-training.json`. Wider-arm training and all
three matched screens remain incomplete. The CPU FP32 equivalence certificate
is not a bitwise BF16 guarantee: initial GPU-BF16 policy CE is 2.3650940 small
versus 2.3653944 wide, value MSE 0.0844391 versus 0.0844800. Different channel
reductions and BF16 rounding produce small initial numerical differences, which
are recorded rather than retuned away. This single-seed, short continuation does
not settle training larger models from scratch or their eventual on-policy RL.

## Both adaptation arms completed

Both models completed 2 epochs / 9,442 GPU updates and selected epoch 1.
Selected validation CE/value MSE: small 2.3588523/0.0842577, wide
2.3597666/0.0843452. This short single-seed adaptation has not established a
predictive benefit for the wider model. The entire curves, initial metrics,
selection metadata and model digests are in
`results/20261026-residual-width-training.json`. The three fixed strength
screens remain pending, with wide-versus-small now running.

Forward-only FP32 timings are archived in
`results/20261026-residual-width-inference-cost.json`. Wall medians at batch32
were 4.784ms small and 5.060ms wide. The batch1 measurements were 6.660ms and
4.942ms respectively. Sequential model order, CUDA algorithm choices and shared
load preclude a general speed ranking; these measurements exclude encoding,
belief construction and search. Actual-clock games remain the strength measure.

The wide-versus-small 64-game screen completed at 35W/3D/26L, 57.03125% score,
paired bootstrap 95% [50.0%,64.84375%] and conservative Hoeffding 95%
[33.02318%,81.03932%]. All 3,036 moves/clocks replayed cleanly, no technical games,
voids or rounding ambiguities. This does not establish promotion or a reliable
capacity benefit. The two teacher screens remain pending. Evidence:
`results/20261026-residual-width-wide-vs-small.json` and its `-audit.json`.

## Closed first capacity campaign

All three fixed screens completed: wide/small35W/3D/26L, small/teacher0W/0D/64L,
wide/teacher1W/0D/63L. All8,531 moves and actual clocks replayed cleanly, with no
technical games, voids or rounding ambiguities. No model is promoted. Two epochs
of this single-seed adaptation do not establish a capacity benefit; these results
do not settle larger-data or longer RL training. Completed campaign/audits:
`results/20261026-residual-width-evaluation.json` and
`results/20261026-residual-width-teacher-screens.json`.
