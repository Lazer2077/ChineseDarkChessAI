# Select a small strong suite with opponent-only screening

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
cannot count as evidence of weakness. The four screens are currently pending.

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
