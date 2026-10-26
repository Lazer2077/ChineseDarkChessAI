# Active research goal

Continuously develop and train ChineseDarkChessAI on CS01 until we can attempt
reproducible matches against every publicly discoverable, runnable full-board
Jieqi AI, and seek statistically supported advantages against all compatible
independent baselines. This is an open research objective, not a guaranteed outcome.

Inventory: `opponents.json`. Include distinct engine variants, deduplicate GUIs
and identical wrappers, and keep incompatible/unavailable entries visible.
Pin sources, weights, commands, rules and budgets. Never classify an untested
engine as defeated. Recheck discovery before claiming complete coverage.

## Current priorities
1. Recover first pilot training after explicitly fixing integer feature dtype.
2. Establish safe UCI adapters and independently inspect referee rule differences.
3. Build/test PikaJieQi, jqe, AB-JChess, musesfish and universe-st baselines.
4. Measure current agent's deficits before replacing the linear pilot with a
   native search and observation/history-conditioned neural evaluation.
5. Search/training ablations, held-out pair matches and promotion gates.

## Evidence gate
A single win is an attempted/successful game, not a strength claim. Formal superiority
requires a predeclared held-out match, at least 100 paired layouts, equal clock and
threads, multiple training seeds, full technical failure counts, and a paired 95%
confidence interval whose lower bound exceeds 50% score. Confirm on another layout
set before promotion. Report results separately for each rule/clock configuration.

Do not weaken an opponent, remove its network, silently patch its search, or
substitute its GUI for its actual engine. Record build patches and hashes explicitly.
No external challenge submissions or messages are authorized by this goal.
