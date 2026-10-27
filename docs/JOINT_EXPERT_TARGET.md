# Matched policy/value supervision factorial

The completed policy-label substitution beats its degraded matched control
51W3D10L, but loses23W4D37Lto warm and1W0D63Lto the unchanged native teacher.
All9309moves pass clean native replay and clock audits. Separately, preserving
warm value during inference yields31W3D30Lin its first direct contrast, with a
time-forfeit and clock ambiguity. These findings do not establish a warm/teacher
improvement or prove that restoring an older value function solves the problem.

The next hypothesis concerns the supervision itself. Native expert actions
and weak-learner terminal returns describe different continuations. Complete
a2x2factorial on exactly the same accepted learner trajectories:

| Profile | Policy target | Value target |
| --- | --- | --- |
| A, already fitted | Original completed search visits | Original learner terminal return |
| B, already fitted | Observer-expert chosen action | Original learner terminal return |
| C, new | Original completed search visits | Observer-expert search-score proxy |
| D, new | Observer-expert chosen action | Observer-expert search-score proxy |

## What the new value target means

The frozen expert receives only the actor's full legal observation history,
public reveals and own private captures. Its already recorded completed cp
score maps to tanh(cp/860); a nonzero mate score maps to its sign. The scale is
fixed from the existing score convention. Missing/invalid/mate-zero scores are
errors, not grounds for silent row deletion or invented terminal values.

This fits a surrogate conditional prediction of that observer-expert search
procedure from the existing history-omitting encoder. It is **not** calibrated
win probability, the current learner's true return, a clairvoyant teacher,
an unbiased asymmetric critic or an on-policy policy-gradient update. Root
native/source legal-set agreement does not equate all counterfactual future
rules or policies. No claim follows from proxy MSE alone.

[Reanalyse/MuZero Unplugged](https://arxiv.org/html/2104.06294v1) motivates
recomputing supervision at stored states and separating policy/value effects;
its Section4and AppendixC discuss trajectory dependence and different value
targets. We use an external native search, not MuZero's learned dynamics or
a proof that one target is universally better. This is an attributed controlled
distillation study; it does not itself constitute a new RL algorithm.

## Fixed implementation controls and recipe

Keep249TRAIN games/13,304positions and31selection games/1,677positions. The same
eight whole source episodes remain excluded for rule incompatibility. No new
rollouts, privileged inputs, actor/history encoder, architecture, phase gate,
inference call, optimizer or augmentation changes are introduced. The fixed
last fourth epoch is exported; profile-specific CE/MSE curves cannot choose a
different checkpoint or duration. Value metrics are named `value_target_mse`
so proxy fitting is not mislabeled as terminal-return prediction.

Use one training seed20261027, warm4ff3c675initialization, four epochs,
batch128, AdamWlr1e-5/weight decay0.0001, clipping1 and416updates per arm.
Three data tests verify exact original-loader controls for both policy targets,
value-only substitution with unchanged observations/policies, actor sign and
fixed scale. The one-epoch/three-update CPU controls exactly reproduce the
earlier `3aa9ffd0` and `3be130a8` checkpoints; both new proxy combinations also
complete finite fitting. These use six original TRAIN games and are integration
evidence only, not isolated generalization.

Before C/Dfit, the full Bterminal-value profile must exactly reproduce its
already selected `0fa5152fd4f4a59a93625579624daf642b9a9b78751f206723a0cdeec94404f8`
checkpoint over all416updates. Failure stops the pipeline. This repetition
is a software/control check under the same training seed, not multi-seed
robustness or another strength result. C/Dthen fit sequentially on one GPU.

## Frozen fresh matches

Run four64-game contrasts: DvsBat seed9660001, Dvswarm at9661001,
Dvsunchanged native teacher at9662001, and CvsAat9660001. The two direct value
replacements intentionally use the same32paired layouts; their effects are
correlated and are not two independent confirmation sets. The warm/teacher
screens use different layout streams. All four finish regardless of earlier
results, without target, scale, duration or checkpoint tuning on matches.

Use actual10+0.1, ordinary one-network inference, LeafBatch32for neural actors,
one CPU thread each, two concurrent games and a shared dedicated inference GPU
within the campaign. No equal-work/FLOP/Hash or simulation-count claim. Require
native identity, legality, terminal and clock audits plus raw archives; preserve
technical failures and timing ambiguity. Beating either degraded terminal-value
control alone is insufficient for warm/teacher progress or promotion. The
final independent multi-set>70%strong-suite/report gate remains unmet.

Evidence: `results/joint-expert-target-research-protocol.json`,
`results/joint-expert-target-integration-pipeline-plan.json`,
`results/20261027-joint-expert-target-integration-*.json` and
`results/joint-expert-target-training-pipeline-plan.json`.

## Completed fitting; fixed matches started

The complete416-update terminal/expert control exactly reproduces0fa5152f,
so the new loader/metric labels do not silently change that reference. Both
new profiles also complete416updates and export epoch4:

| Profile | Model NPZ SHA256 | Selection proxy MSE, initial → final |
| --- | --- | --- |
| C, search policy/proxy value | `aa2b6b05190942b196b79edcfc37226e75bf84ed7bc8bd9808fc2f9f4d30cc5c` | 0.064631 → 0.065103 |
| D, expert policy/proxy value | `aaebd5a81326b995990c68bfbbe3d5f75e9c993c060bb38767233b46bc998eaf` | 0.064631 → 0.058358 |

These selection losses describe the fixed surrogate on reused data. Lower
proxy error does not establish better game values or wins; equal loss weights
also do not equate gradient norms or target noise across terminal and proxy
supervision. All four prespecified matches have been queued with the frozen
artifacts. Evidence:
`results/20261027-joint-expert-target-{terminal-expert-reproduction,proxy-search,proxy-expert}-training.json`
and `results/joint-expert-target-evaluation-plan.json`.

The first direct value-target replacement at expert policy completes39W4D21L/64,
64.0625%score, paired bootstrap95%[0.53125,0.7421875]. All3307moves pass native
reveal, legality, terminal and clock audits without technical games or timing
ambiguity. This improves over the terminal-value/expert-policy control only;
warm/teacher superiority remains unestablished and the other prespecified
contrasts continue unchanged. No promotion follows from this result. Evidence:
`results/20261027-joint-expert-target-proxy-expert-vs-terminal-expert{,-audit}.json`.

The independent-layout warm-model screen completes32W2D30L/64,
51.5625%score, paired bootstrap95%[0.3984375,0.6328125]. All3260moves pass
the native identity, legality, terminal and clock audits, with zero technical
games or timing ambiguity. The positive degraded-control result therefore
does not establish an improvement over the warm model. The native-teacher and
search-policy factorial contrasts continue as prespecified, without retuning.
Evidence: `results/20261027-joint-expert-target-proxy-expert-vs-warm{,-audit}.json`.
