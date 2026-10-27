# Select a small strong suite with opponent-only screening

Targeted2026-10-27refresh finds no newly established compatible independent
engine: AB's latest release remainsv0.2d, the jqe teacher main commit is unchanged,
and official Pikafish advertises the already pinnedjieqi_oldbranch, with no
currentjieqibranch. This is a targeted refresh, not exhaustive discovery;
`results/20261027-opponent-refresh.json` records queries and primary endpoints.

Three pending candidates now have pinned complete-tree/source availability
audits. BUPT's116-file Java project uses an incompatible ignore-check/six-check-
chase profile. zhzjzz's41-file Java project permits in-place dark flips and has a
local search fallback plus optional hosted-model selection. Lazyy's pinned tree
contains only a README. None has an established license grant in the inspected
evidence, and none is newly runnable/compatible or defeated by this audit.
No API calls or builds were performed. Original private-capture/terminal and
runtime questions remain scoped explicitly in
`results/20261027-pending-opponent-source-audit.json`. These findings do not
change the provisional three-profile strong suite or any strength result.

## Broader source discovery, 2026-10-27

Six newly pinned source audits expand the ongoing inventory to46entries.
This is a registry update, not46independent engines, a full refresh of every
older entry, exhaustive discovery or any new defeat. The strong suite remains
unchanged; no newly compatible independent strong baseline is established.

| Candidate | Verified evidence | Remaining limit |
| --- | --- | --- |
| miaosiSari/JieqiMCTS | GPLv3 root license; unchanged C++ sources build with installed binding headers; callable AIBoard4 hints and opening44moves matching native | MCTS tree scaffolding is not an established complete MCTS playing controller. A fully public check fixture has15God-generated moves versus2native legal moves; its depth1hint is nevertheless legal. Strict playing adapter, full history/private information and independent strength remain unverified. |
| rexfang0829/jieqi-ai | Complete pinned tree has only README and project plan | No runnable game/search/model; no license grant established in inspected evidence |
| RemiliaScarlet-57/masked-chinese-chess | TypeScript board, probability and alpha-beta source | Grant, private-capture view and full terminal profile unverified. Hypothetical-state code independently samples dark types and marks all revealed; this approximate search model is not proof of referee compatibility. |
| ZJQJZ/chess_z | Only advertised master reviewed; ordinary Xiangqi README, seven-piece enum and ordinary position interface | Full Jieqi mode and license grant not established; repository description alone is insufficient |
| aSun-King-W/chess | Multi-mode frontend, Pikafish WASM integration and local heuristic fallback described | Jieqi AI path/rules/observation and project grant unverified; integrated engine is not automatically another independent family |
| hjfkj/dsh-xiangqi | MIT root license and self-contained alpha-beta implementation described | README explicitly discloses true dark identities in search and missing long-check/chase/no-capture outcomes; default profile is not a fair strict-observation baseline |

The C++/MCTS build substitutes only binding headers for the current Python ABI;
game/search sources remain unchanged. Its opening smoke and fully public
in-check probe are feasibility/differential evidence, not matches or a claim
that every hint is illegal. Do not discard all potential callable C++ search
variants merely because a complete MCTS loop is unestablished. A future
compatible profile still needs explicit observation, history, terminal, clock
and opponent-only strength checks before suite admission.

Pinned primary URLs, complete-tree availability, reviewed files, build digest
and bounded conclusions are in
`results/20261027-broader-opponent-source-audit.json`,
`results/20261027-jieqi-mcts-build-smoke.json` and
`results/20261027-jieqi-mcts-public-check-probe.json`.

The user authorized opponent-versus-opponent preselection to reduce repeated
candidate evaluation on 2026-10-26. The final routine suite may contain only a
few profiles, ordinarily about 3–5. Its declared strong members must each meet
the separate >70%-win confirmation/report gate. A broad discovery inventory is
still retained, but direct candidate matches against every project are no longer
a completion prerequisite. Single-training-seed learning is allowed; independent
multi-test-seed confirmation remains required. See `GOAL.md` for both amendments.

## First stage: use a common anchor

Avoid a full round robin initially. Two already completed, clean 200-game sets
of the unchanged original jqe+it014 teacher versus released AB-JChess v0.2d give
149W/4D/47L and 157W/4D/39L from the teacher's perspective, actual 5+0.05,
one thread each and native Hash64MB. Existing paired/identity/legal/clock evidence
is reused, rather than rerunning 400 games. These are original open-source
opponents, not results for our trained candidate. The teacher remains an
attribution control regardless of screening rank.

Evidence: `results/20261026-abj-teacher-{a,b}.json` and their `-audit.json` files.
The new plan pins those reports and audits, the exact AB binary and matching
official V11 network, sources/adapters and referee. It keeps the same external
rule/time stratum and fixes these four new screens before outcomes:

| Left anchor | Right opponent profile | Games / pairs | Layout seed |
| --- | --- | --- | --- |
| AB-JChess v0.2d | Native build of brianhliou/Pika WASM classical branch | 40 / 20 | 8610001 |
| AB-JChess v0.2d | universe-st/Jieqi hard search | 40 / 20 | 8620001 |
| AB-JChess v0.2d | Musesfish native-public-root/timed profile | 40 / 20 | 8630001 |
| AB-JChess v0.2d | Yuan pure root-consistent ISMCTS | 40 / 20 | 8640001 |

This costs 160 new games instead of an immediate 600-game round robin over six
engines. Both players are external projects. Shared external rules are native-jqe
80 no-capture plies and repetition/check/chase responsibility, with full
observer-private history. Internal rule approximations still differ by profile.
All games use actual 5+0.05, no timing grace, one search thread each and two games
concurrently; no GPU is used. Native Hash64MB does not apply to Python/TypeScript
opponents that ignore Hash and use their own structures. No equal-FLOP or memory
claim is made. The strict/padded serialization is selected separately for each
engine; originals and their official networks are not weakened or replaced.

Recipe: `results/baseline-anchor-screen-plan.json`. Runner:
`scripts/screen_baselines.py`. Node-runtime digest is recorded without a private
runtime path. Each match independently archives W/D/L, whole-layout bootstrap
and conservative Hoeffding intervals, voids and technical outcomes, followed by
seeded identity/legal/terminal/actual-clock replay. Engine identities are mapped
to the referee's generic candidate/baseline labels only for serialization;
neither is our learned model. Unclean/ambiguous screens remain unverified and
cannot count as evidence of weakness. All four screens have completed; see the provisional suite below.

## Selection and claims

Use clean anchor results to prioritize a provisional shortlist, not to assert a
complete rank or transitive victory. The intervals from small screening sets are
exploratory and multiple comparisons affect selection; final model confirmation
must use different seeds and frozen checkpoints. No engine is automatically
removed by a single loss or a rounded-clock/process failure.

Retain the teacher, strong independent competitors, and a small number of
distinct-style or potentially nontransitive contenders. If an anchor comparison
is inconclusive or two candidates contest a place, spend additional paired games
on that place instead of expanding every pair. Different rule profiles require
separate strata; unavailable intended weights and unresolved runtime failures
are coverage gaps, not weak-engine scores. Record explicit inclusion/exclusion
reasons and final versions before using our candidate's confirmation results.

Screened-out projects remain in `docs/opponents.json`; exclusion never becomes a
claim that our model beat them. New relevant strong runnable versions trigger
selection review. The eventual report must state its shortlisted-suite scope
and inventory cutoff, rather than claiming victory over every discovered project.

## First completed anchor results

AB versus Pika completed 23W/0D/17L in 40 games; paired bootstrap 95% [42.5%,72.5%]
cannot rank them confidently. All2,605 moves/terminals replayed correctly, with
no technical games or voids. The original rounded-clock audit marks 22 ambiguous
checks, so the frozen controller records this screen ineligible. A separate
[high-resolution outer-clock witness](PRECISE_CLOCK_WITNESS.md) certifies wall
safety using the existing timing log, minimum remaining lower bound 2.02679ms.
The original eligibility flag is preserved; no engine is eliminated from this
inconclusive comparison.

AB versus Universe completed 39W/0D/1L, all1,477 moves/clocks cleanly replayed
(the exact count remains in the independent audit). It is eligible for screening,
not a transitive win by our model. The complete four-screen selection is below. Results:
`results/20261026-baseline-screen-abj-vs-{pika-wasm-native,universe-hard}.json`
and matching `-audit.json` files.

## Provisional three-profile routine suite

All four fixed anchor screens completed (160 new games): AB versus Pika23/0/17,
Universe39/0/1, Musesfish37/0/3, Yuan pure40/0/0. The latter three screens are
cleanly eligible under their original audits. All 7,259 played moves were legally
replayed; Pika's supplemental timing certificate and original ineligibility are
reported separately above. These are opponent-only results, not new wins by our
learner. The remaining anchor screens are no longer pending.

The provisional routine suite is now:

| Profile | Why retain it |
| --- | --- |
| Original jqe+it014 teacher | Strong direct anchor evidence and mandatory attribution control |
| AB-JChess v0.2d + matching official V11 | Strong independent implementation relative to jqe, verified released weights |
| Native classical Pika WASM branch | Uncertain ordering against AB; keep the distinct classical-evaluation variant |

AB and classical Pika share broader Stockfish/Pikafish ancestry; three profiles
are not three wholly independent algorithm families. Universe hard, the timed
Musesfish profile and Yuan pure are removed from routine candidate evaluation
by direct anchor evidence, while staying in the inventory and diagnostic record.
This does not prove dominance at other clocks, rule profiles or versions.
Newer neural Pika candidates, unresolved weights/runtime failures and separate
rule profiles are not classified as weak. Refresh discovery and resolve relevant
new runnable contenders before fixing the final report suite.

Exact selection reasons, artifact hashes and all direct evidence are retained in
`results/20261026-baseline-preselection.json`. This is a provisional resource
allocation decision, independent of the outcome of our candidate's final tests.
The >70%-win gate still requires the same frozen learned model to pass fresh,
independent test-layout sets against every final declared strong baseline.
