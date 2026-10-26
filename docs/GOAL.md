# Active research goal

Continuously develop and train ChineseDarkChessAI until we can attempt
reproducible matches against every publicly discoverable, runnable full-board
Jieqi AI, and seek statistically supported advantages against all compatible
independent baselines. This is an open research objective, not a guaranteed outcome.

The user extended the objective on 2026-10-26: after stable win rates above 70%
against all current open-source/research strong baselines are established,
write an arXiv-style technical report with training curves, the development
process, our core innovations and meaningful implementation details. This
requirement is part of the active goal and all subsequent completion audits.

Inventory: `docs/opponents.json`. Include distinct engine variants, deduplicate GUIs
and identical wrappers, and keep incompatible/unavailable entries visible.
Pin sources, weights, commands, rules and budgets. Never classify an untested
engine as defeated. Recheck discovery before claiming complete coverage.

## Current priorities
1. Complete the fixed current-actor neural self-play policy/value series, recording real visit targets, terminal rewards, curves and held-out paired diagnostics.
2. Diagnose the neural search deficit against the unchanged teacher, including GPU inference cost and search depth. Retain completed negative NNUE, capacity and phase controls.
3. Extend fair original-engine coverage with complete observer-specific histories.
4. Exploit declining future-relevant identity uncertainty: distinguish actor/public certainty, test full-public belief abstraction, entropy-conditioned computation and deterministic-continuation curricula. Investigate belief/history representation and search-target bias when learning stalls. Model scaling is also explicitly authorized when win rate stalls: test wider/deeper residual networks on GPUs, with matched data/initialization controls, inference-cost measurements and equal actual-clock strength evaluation; see [capacity protocol](MODEL_SCALING.md).
5. Verify independent held-out confirmation before any strength or coverage claim.

## Communication and public research record
Notify and push only for substantial progress, a meaningful method/result change,
or an issue requiring user action. Do not send unchanged-status notifications.
When a method stalls, consult primary papers on established chess/game AI and
form testable hypotheses adapted to Jieqi's partial observability and private
information. Before the next push, keep README focused on academic methods,
experimental evidence and limitations. Exclude host identities, usernames and
local absolute environment paths from public research documentation.

## Training and test seeds: user amendment, 2026-10-26

A single training seed is sufficient; successful replication across training seeds
is not an acceptance or report prerequisite. Prioritize one training run and use
multiple independent, predeclared held-out test/layout seeds for confirmation.
Freeze the selected checkpoint before these tests and use the same checkpoint
across the required sets. Select checkpoints on separate validation data; do not
select among checkpoints or retune using confirmation outcomes. If a model is
changed after testing, confirm the new model on fresh held-out test seeds.

Retain training seeds and completed multi-seed experiments for reproducibility.
Additional training-seed replications are optional when a concrete diagnostic
requires them, not automatic work to satisfy the goal. Single-training-seed
results support the tested checkpoint under the stated conditions; they do not
establish robustness across training initialization. Report that scope honestly.
This amendment supersedes older prose requiring multiple successful training
seeds. Archived experiments/plans keep their original criteria and meaning;
removing the training-seed prerequisite does not turn failed tests into passes.

## Evidence gate
A single win is an attempted/successful game, not a strength claim. Formal superiority
requires a predeclared held-out match, at least 100 paired layouts, equal clock and
threads, multiple independent test/layout seeds, full technical failure counts, and a paired 95%
confidence interval whose lower bound exceeds 50% score. Confirm on another layout
set before promotion. Report results separately for each rule/clock configuration.

Do not weaken an opponent, remove its network, silently patch its search, or
substitute its GUI for its actual engine. Record build patches and hashes explicitly.
No external challenge submissions or messages are authorized by this goal.

## Strong-baseline 70% gate and technical report

Refresh and date the discovery inventory before final confirmation. Identify
strong open-source and research baselines using available source/model artifacts,
algorithmic independence and reproducible comparative evidence. Include the
unchanged upstream teacher, relevant current AB-JChess/neural Pika versions when
available, and independently implemented research methods with compatible rules.
Do not substitute obsolete/easier variants for a relevant strong version.
Unknown or unavailable strong candidates remain explicit coverage gaps; missing
weights cannot be recorded as wins or silently removed to declare universal
coverage. Paper claims without a runnable implementation are literature context,
not measured victories. Record the final coverage cutoff and unresolved entries.

"Win rate" means W/(W+D+L), separately from score (W+0.5D)/N. For each eligible
strong baseline, require more than 70% wins on each of at least two disjoint,
predeclared held-out color-swapped layout sets. A stable-above-70% claim also
requires a paired, layout-level 95% lower confidence bound above 70%, clean
technical audits and confirmation across independent test/layout seeds using the
same frozen checkpoint. Multiple successful training seeds are not required. Choose sample sizes
before outcomes for adequate precision/power; 100 pairs is only a minimum.
Pin budgets, source/model hashes and rule profiles. Disclose GPU/CPU resource
asymmetry and compare learned variants with matched resources. Do not pool weak
opponents, forks or clocks into a universal 70% statement. Existing 50%-score
acceptance artifacts keep their original meaning and do not satisfy this gate.

Once the full gate is met, deliver an English arXiv-style report with compilable
LaTeX, bibliography, PDF and reproducible figures, including:

- Abstract, related work, precise Jieqi/private-information rules and assumptions.
- Observation/belief model, declining uncertainty, search/training objectives and
  pseudocode; distinguish inherited methods from measured original contributions.
- Actual training/self-play curves, steps/games/compute, the actual training seed
  and multiple independent test seeds, uncertainty intervals and checkpoint selection. Preserve negative trials and corrected-data
  provenance; do not invent smooth curves or missing measurements.
- Fixed-version strong-baseline W/D/L tables, paired intervals, rule/time/resource
  profiles, coverage accounting and technical audits.
- Ablations for claimed innovations, including fully public continuation; discuss
  approximate beliefs/signaling and limits beyond tested opponents and clocks.
- Implementation details affecting correctness/performance: observer isolation,
  private captures, chance sampling, visit targets, batched inference, phase
  transitions and reproducible seeds. Exclude machine identities, usernames and
  private absolute environment paths.

Retain raw logs and figure provenance now. Earlier data preparation/report
structure is allowed; the result and report trigger remain unachieved until
the full gate passes. arXiv style specifies format, not authorization to submit
to an external publication service. The goal remains active until the research
evidence and the requested report are both complete.

## Confirmed scope so far

Three fixed opponent versions meet their original two-set criteria under the documented
external referee/clock profiles; see [confirmed results](CONFIRMED_RESULTS.md).
This is partial coverage, not completion. Direct improvement over the upstream
jqe teacher remains unconfirmed. Inventory entries with missing weights,
incompatible rules, unresolved licenses or technical failures stay unverified.
