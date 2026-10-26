# Diagnose normalization state during small-data neural RL

The first two fixed neural RL updates lost to their preceding actors at
24W/1D/39L and 17W/6D/41L. Learning loss alone did not predict stronger play.
Besides noisy visit targets and weak search, normalization statistics may change
the network function when adapting the warm model to a different trajectory
distribution. This is a testable hypothesis, not an established cause.

## Inference-only factorial

The fixed `results/normalization-factorial-plan.json` combines the warm model's
and first RL checkpoint's learned weights with their running BatchNorm buffers.
Weights include affine gamma/beta; only running means, variances and tracked
batch counters are transplanted. All 99 buffer sources and every remaining
tensor are checked exactly. Two tests cover tensor provenance, unchanged affine
parameters/RNG, identical endpoints and incompatible architectures.

The four combinations are evaluated on the same 32 existing checkpoint-selection
episodes, 1,758 positions, using CPU FP32 and no optimizer updates. The trained
control reproduces its archived GPU selection losses within 1e-4. These are
already-used selection trajectories, not fresh strength evidence.

| Weights | Running buffers | Visit-policy CE | Terminal-value MSE |
| --- | --- | --- | --- |
| Warm | Warm | 5.574549 | 0.759373 |
| RL | RL | 3.877386 | 0.760621 |
| RL | Warm | 3.849723 | 1.040608 |
| Warm | RL | 6.728530 | 0.920476 |

Restoring old buffers after updating weights does not fix the terminal-value
selection error in this intervention. Transplanting new buffers into old weights
also worsens these losses. Weights and statistics co-adapt, so this is not a
controlled frozen-statistics training experiment and cannot exclude such a method.
Visit targets include exploration; their CE is not playing strength. Full curves,
state hashes and output shifts: `results/20261026-normalization-factorial.json`.

## Controlled frozen-statistics training

A separate fixed two-arm update now uses the first iteration's exact 256 training
and 32 selection games, initial checkpoint, augmentation stream, four epochs,
batch128, learning rate1e-5 and 436 updates. Both use the unchanged original RL
trainer. The control uses its ordinary training mode. The second arm keeps every
BatchNorm running buffer at its warm-start value during training while affine
and all other network parameters remain trainable. Inference uses those same
fixed buffers. This differs from restoring old buffers after training.

`research/frozen_normalization.py` changes only the loaded model instance's
training-mode behavior. No original class, source or active training process is
modified. A targeted test verifies fixed buffers, unchanged initial outputs,
nonzero affine gradients, real weight updates and isolation from other instances.
Every training/evaluation mode transition checks the buffers, and a post-training
proof lists any changed buffer. Safe exported arrays use the ordinary inference
architecture; they do not require deserializing a custom Python model object.

Recipes: `results/frozen-bn-rl-{control,frozen}-plan.json` and
`results/frozen-bn-rl-series-plan.json`. The two arms train sequentially on one
otherwise free GPU, separately from all frozen existing series. They use one
training seed; any future strength claim needs fresh paired test seeds. No
promotion follows from a lower selection loss or unchanged buffers.

Both arms completed436updates and selected epoch4. The new control checkpoint
bitwise reproduces the original RL checkpoint, SHA256
`168ae3d997d8c671acf91d36b73d23eeedfdfc48985fc2f6a91948526b4d7d98`.
Frozen-statistics selection CE/MSE is3.6332916/0.7634463 versus control
3.8773666/0.7605823; no frozen buffer changed and affine parameters remain
trainable. The CE improvement does not establish stronger play. Exact curves,
proofs and selected digests: `results/20261026-frozen-bn-rl-training.json`.

The fixed `results/frozen-bn-evaluation-plan.json` now tests frozen statistics
against the original warm model on64fresh games (seed8950001), then the unchanged
native teacher on64fresh games (seed8951001), actual10+0.1, leaf batch32. Both
neural actors share one GPU and one CPU thread each; the native teacher is CPU
NNUE, so no equal-total-compute claim is made. Complete private-history/native
legal/terminal and clock audits follow each screen. The already negative joint-
update control is not substituted for the stronger original warm model in this
strength test. All checkpoints were selected on the old validation split before
these fresh tests. Both screens are currently running/pending; no promotion.
