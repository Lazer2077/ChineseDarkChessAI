# Active research goal

Continuously develop and train ChineseDarkChessAI until we can attempt
reproducible matches against every publicly discoverable, runnable full-board
Jieqi AI, and seek statistically supported advantages against all compatible
independent baselines. This is an open research objective, not a guaranteed outcome.

Inventory: `opponents.json`. Include distinct engine variants, deduplicate GUIs
and identical wrappers, and keep incompatible/unavailable entries visible.
Pin sources, weights, commands, rules and budgets. Never classify an untested
engine as defeated. Recheck discovery before claiming complete coverage.

## Current priorities
1. Complete the frozen multi-seed selfplay/value-learning series and paired gates.
2. Test same-data capacity controls at equal actual clock, retaining negative results.
3. Extend fair original-engine coverage with complete observer-specific histories.
4. Exploit declining future-relevant identity uncertainty: distinguish actor/public certainty, test full-public belief abstraction, entropy-conditioned computation and deterministic-continuation curricula. Investigate belief/history representation and search-target bias when learning stalls.
5. Verify independent held-out confirmation before any strength or coverage claim.

## Communication and public research record
Notify and push only for substantial progress, a meaningful method/result change,
or an issue requiring user action. Do not send unchanged-status notifications.
When a method stalls, consult primary papers on established chess/game AI and
form testable hypotheses adapted to Jieqi's partial observability and private
information. Before the next push, keep README focused on academic methods,
experimental evidence and limitations. Exclude host identities, usernames and
local absolute environment paths from public research documentation.

## Evidence gate
A single win is an attempted/successful game, not a strength claim. Formal superiority
requires a predeclared held-out match, at least 100 paired layouts, equal clock and
threads, multiple training seeds, full technical failure counts, and a paired 95%
confidence interval whose lower bound exceeds 50% score. Confirm on another layout
set before promotion. Report results separately for each rule/clock configuration.

Do not weaken an opponent, remove its network, silently patch its search, or
substitute its GUI for its actual engine. Record build patches and hashes explicitly.
No external challenge submissions or messages are authorized by this goal.

## Confirmed scope so far

Two fixed opponent versions meet their two-set criteria under the documented
external referee/clock profiles; see [confirmed results](CONFIRMED_RESULTS.md).
This is partial coverage, not completion. Direct improvement over the upstream
jqe teacher remains unconfirmed. Inventory entries with missing weights,
incompatible rules, unresolved licenses or technical failures stay unverified.
