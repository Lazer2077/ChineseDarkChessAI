# Larger relative-decoder backbone on the same teacher replay

The frozen relative128x15pretrained actor improves over the old neural reference
in64development and200fresh confirmation games, but scores0/2/62against the
unchanged native teacher. This motivates one bounded capacity test, alongside
separate rule-label and policy-target controls. More parameters are not an
innovation or a guarantee that capacity is the remaining bottleneck.

[KataGo2019section4.6](https://arxiv.org/html/1902.10565v1) describes progressive
network sizing and fitting a larger network on the same replay. We borrow that
experimental motivation, without importing its Go results or replacing actual
Jieqi strength tests with predictive accuracy.

## Frozen comparison

| Model | Decoder | Channels | Blocks | Parameters |
| --- | --- | ---: | ---: | ---: |
| Completed reference | Relative50 | 128 | 15 | 5,453,555 |
| New capacity fit | Relative50 | 256 | 20 | 24,918,259 |

The candidate is cold initialized with the same single seed20261029as the
reference. This is not function-preserving growth or a width-only effect:
width, depth, initialization function and compute differ. The relative50mapping,
ordinary value head and observer encoder remain unchanged. Both have phase
experts disabled; actual both-dark-zero inventory canonicalization remains.
No true hidden identity, opponent-private capture, fabricated history plane or
new information-state sufficiency assumption is introduced.

Use exactly the original four certified replay files:2,417,541searched positions
from32,768generated games, plus the same512selection games/37,471positions.
Retain one-hot actual teacher choices and0.7*tanh(score/860)+0.3terminal returns.
Fit all12epochs at batch512and56,652updates with the same AdamWlr0.0003,
weight decay0.0001, gradient clip1,5%warmup/cosine schedule, BF16when supported
and dedicated example/mirror RNG stream. There is one training seed and one
training device, not distributed training or equal FLOPs. Reuse of the selection
set means its metrics are not an independent generalization estimate.

Finish all12epochs and select the minimum common selection CE+MSE, excluding
initialization. No early stopping or match-based checkpoint choice. The original
relative parent remains frozen as
`e43ff2aaf276fe8d5700da12c0ca00f1bdd9882f676dc1c51c3ed809dac28c1f`.
Candidate initial NPZ:
`257b3d684223f02828ca79e72abeb36b220ac0b1dd53ce4af9df4b5a527c725e`.

The reused initialization helper also constructs a dense256x20model to check
common arrays/value outputs and codec consistency. That auxiliary dense model
is not another fitted arm or an extra strength control in this study.

## Full-size integration completed; full replay fit running

Before full fitting, a512-row TRAIN batch completes one real AMP update with
finite loss/gradients. The separate scratch weights round-trip exactly through
numeric NPZ; the planned cold initial file remains unchanged. A bounded opening
search completes8simulations and chooses a legal move. Peak allocated memory is
2,224,753,152bytes. This checks memory, training arithmetic and integration,
not generalization or full-game reliability.

The same preflight measures eagerFP32encoding+forward+CPU-return wall time on
TRAIN inputs, three warmups and ten local repeats per batch:

| Batch | Reference128x15, ms | Candidate256x20, ms |
| --- | ---: | ---: |
| 1 | 5.450 | 6.505 |
| 8 | 5.656 | 6.426 |
| 32 | 5.607 | 7.341 |

The reference is trained and the candidate has one scratch update; this is an
architecture-cost check only. These are local pipeline timings, not GPU-kernel
times, full-search speedups or an actual-clock strength result. Candidate full
fitting starts from the untouched cold initialization, not the scratch update.

## Required subsequent strength tests

After the prescribed selection, freeze artifacts before both64-game development
screens: candidate versus relative128x15parent at9770001and versus unchanged
native teacher at9771001. Use actual10+0.1, LeafBatch32for neural actors, one CPU
thread each and two concurrent games sharing an inference device. Complete both
regardless of results and retain native identity/legal/terminal/clock audits and
all technical failures. A favorable predictive metric, parameter count or
fixed-work result cannot establish equal-time strength.

Data coverage, history omissions and search approximation may remain limiting
even if capacity helps prediction. No strong-teacher, final-suite or arXiv-report
claim is supported by the currently running fit.

Evidence: `results/spatial-scale-research-protocol.json`,
`results/spatial-scale-initialization-plan.json`,
`results/spatial-scale-preflight-plan.json`,
`results/20261027-spatial-scale-preflight.json`,
`results/spatial-scale-training-plan.json` and
`results/spatial-scale-training-pipeline-plan.json`.
