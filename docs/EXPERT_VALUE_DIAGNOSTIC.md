# Value-target semantics and preserved-value inference control

The observer-expert policy update replaces only the policy labels. Its value
targets still describe terminal returns of the original weaker learner under
Yuan rules. A stronger policy and a value fitted to that behavior need not form
a coherent improved search policy. This is a hypothesis to test, not a reason
to declare recorded terminal outcomes false or teacher search scores correct.

[MuZero Unplugged/Reanalyse](https://arxiv.org/html/2104.06294v1) distinguishes
policy targets recomputed at stored states from trajectory-dependent value
targets (Section4). AppendixC compares TD targets and direct search-value
regression; neither is universally preferable in its experiments. This motivates
separating our policy/value interventions, not claiming that tanh of an NNUE/PVS
score is an unbiased POMDP return or that we implemented MuZero.

## Frozen descriptive target study

Use exactly31accepted original selection games/1,677positions. These were
already logged during the fixed-duration policy-target fit and are not a new
independent test set. Three actors remain frozen: original warm, matched
visit-target control and observer-expert policy target. No model is fitted,
selected or tuned using this diagnostic.

The teacher score proxy is tanh(cp/860), retaining side-to-move perspective;
nonzero mate scores map to their sign. Missing, invalid or mate-zero scores
are rejected. The fixed scale matches the existing score convention, but does
not calibrate scores into win probabilities. Teacher scores concern native
search with different continuation behavior; source terminal outcomes concern
the recorded weaker learner. Distances to the two targets answer different
questions and their magnitudes are not interchangeable quality measures.

Primary descriptive aggregation below averages MSE within each game, then
across games. Position-weighted metrics, input aliases and observer-only entropy
strata are archived separately.

| Frozen actor | Source learner terminal MSE | Teacher search-proxy MSE |
| --- | ---: | ---: |
| Warm | 0.810011 | 0.061983 |
| Matched visit-target control | 0.812865 | 0.110023 |
| Observer-expert policy target | 0.795867 | 0.095249 |

The expert-policy model is closer to the source outcome labels than the other
two fitted/initial functions in this descriptive sample, while both new value
functions move away from the teacher proxy relative to warm. This does not
establish which value function is more accurate for the deployed policy,
or that restoring warm value improves play. Every phase metric retains its
actual number of positions and games; small strata support no broad conclusion.

## Exact representation aliases

Group the exact board, actor, known inventories and no-capture inputs used by
the current encoder; clear inventories at the same actual-both-dark-zero gate.
Score, result, ply and action labels do not enter the key. Full projected
observer histories retain public reveals and only the current actor's private
captures, independently reconstructed from complete seeded source replay.

This finite sample has1,645distinct input keys and three repeated-input groups.
No repeated key has different projected histories or conflicting expert actions.
Thus this particular exact-duplicate check does not identify hidden history
conflicts as the explanation for failure. It cannot prove history sufficiency:
almost every input is unique, and unsampled/nearby states can still require
different information. The empirical minimum deterministic policy CE is0and
terminal MSE is0.0183123on this finite sample; these are fitted-sample quantities,
not Bayes-risk bounds, generalization estimates or attainable neural performance.

Three targeted tests cover score perspective/ambiguous mates and equivalence
of input keys to the actual encoder under public canonicalization, unchanged
label bytes, different dark inventories, actor and no-capture counts.

## Separate reused teacher-retention diagnostic

The same fixed actors are evaluated on37,471positions from512repeatedly reused
teacher-selection games, distinct from the31learner-selection games above.

| Frozen actor | Teacher action CE | Warm-to-model full-action KL | Old mixed-value MSE |
| --- | ---: | ---: | ---: |
| Warm | 2.364835 | 0 | 0.084464 |
| Matched visit-target control | 2.751456 | 0.535309 | 0.174218 |
| Observer-expert policy target | 2.428072 | 0.121962 | 0.161600 |

Expert policy supervision retains much more of the original teacher policy
than the visit-target control, while value retention remains substantially
different from warm. Warm itself was selected on these teacher-selection data.
These are descriptive retention results, not fresh generalization, legal-only
KL, a component-causality proof or evidence of playing-strength improvement.

## Frozen direct inference intervention

Test the already fitted expert policy with the original warm value function.
Keep both complete networks and their own normalization statistics intact;
do not transplant layers or buffers. Compare it against the same expert policy
with its newly fitted value, original warm policy/value, and unchanged native
teacher. Freeze64games each at seeds9650001,9651001and9652001before any of these
outcomes. Finish all contrasts regardless of earlier results. The incomplete
policy-target campaign's WDL was not inspected to choose this intervention.

All neural candidates and neural baselines execute two complete64-channel,
15-block forwards, including identical-model controls. Use actual10+0.1,
LeafBatch32, one CPU thread per actor, two concurrent games and a dedicated
shared inference GPU for this campaign. The native teacher remains unchanged.
Behavior affects throughput and trajectories, so equal clock is not an exact
fixed-simulation or equal-work causal decomposition. Retain all native replay,
seeded reveal, terminal and clock audits and any technical flags. No training,
privileged inputs, checkpoint reselection or promotion follows from the offline
diagnostics. The direct updated-value contrast alone cannot establish progress
over warm or the teacher, and this pilot cannot meet the final70%report gate.

Evidence: `results/expert-value-target-diagnostic-plan.json`,
`results/20261027-expert-value-target-diagnostic.json`,
`results/20261027-expert-policy-target-retention.json`,
`results/expert-policy-target-retention-plan.json` and
`results/expert-policy-warm-value-evaluation-plan.json`.

The first direct preserved-value contrast completes31W3D30L/64 against the same
expert policy with its updated value,50.78125%score, paired bootstrap95%
[0.3828125,0.640625]. It contains one time-forfeit game and one rounded-clock
ambiguity, so it is not accepted as a clean superiority test. All3547recorded
moves replay legally with matching seeded reveals. Preserve the technical
flags and complete raw sample; do not remove the failed game or infer that warm
value restores playing strength. The prespecified warm/teacher contrasts
continue unchanged. Evidence:
`results/20261027-expert-policy-warm-value-vs-updated-value{,-audit}.json`.

Against original warm policy/value under the same two-forward convention,
the preserved-value profile completes33W2D29L/64,53.125%score, paired
bootstrap95%[0.4453125,0.625]. All3462moves pass native replay and clock audits
without technical games or timing ambiguity. This does not establish warm
superiority. Its different layouts and two-forward cost also prohibit treating
the result as a paired improvement over the ordinary single-forward23W4D37L
policy-target contrast. The unchanged-teacher comparison continues unchanged.
Evidence: `results/20261027-expert-policy-warm-value-vs-warm{,-audit}.json`.
