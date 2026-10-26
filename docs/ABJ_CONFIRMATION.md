# Released AB-JChess comparison: expanded learner and teacher control

The initial four-game movetime smoke used a different candidate/search and is
not pooled with this experiment. The expanded NNUE candidate, with unchanged
jqe native search, completed a new 32-game development match against released
AB-JChess v0.2d plus its official V11 network: 27W/1D/4L, score 85.9375%, 16
complete layout pairs, no technical failures or voids. Small development sets
are not independent confirmation. See results/20261026-expanded-abj-development.json.

This candidate's own 200-game teacher gate was 87W/13D/100L, so the AB result
cannot be described as evidence that training improved the inherited engine.
The published model models/experimental/expanded-seed20261026-it1.jqn remains
unpromoted, with upstream MIT attribution; its hash matches the evaluated net.

## Frozen follow-up

Two held-out sets each contain 200 games / 100 layout pairs. Each has two arms:

- The frozen expanded learner against the released AB engine/network.
- The unchanged published jqe teacher against the same AB engine/network,
  using the same initial layout pairs as the learner arm.

Seeds are 7920001 and 7930001, actual game clock 5+0.05, one thread and 64 MB
per engine, two concurrent games. Networks are preloaded before game clocks.
Both sets and both arms run in full without changing the candidate based on
interim scores. Recipe and source/binary/network hashes are frozen in
results/abj-confirmation-plan.json; the launcher verifies them before each arm.

The original AB search and official release network are retained. Its build
has only the documented GCC11 constexpr declaration repair, with unchanged
function bodies. The default ChasingRule=skyrule_Jieqi and other native search
settings are retained, not tuned to make the opponent easier. Complete
actor-specific histories include public reveals and captor-only identities;
AB uses its strict suffix serialization without classical x padding.

Games use the fixed independent jqe native referee: 80-ply no-capture and
third-public-repetition/check/chase responsibility. AB's internal SkyRule and
jqe's internal cycle search are heuristics relative to that referee. We do not
claim identity of all internal adjudication details, all tournament conventions,
or superiority under other profiles. Any illegal move, timeout, infrastructure
void or incomplete layout pair prevents confirmation acceptance.

Each learner arm must have 100 complete pairs, no technical failures/voids,
and both paired bootstrap and bounded Hoeffding 95% lower score limits above
50%; repeat on the second set. Teacher-control WDL stays separate. An advantage
over AB may come largely from the inherited jqe architecture/weights; a claim
of a learned improvement requires its own evidence. One training seed and one
clock/profile do not establish the full all-engine objective.

## First held-out learner set completed

Set A finished 162W/7D/31L across all 200 games / 100 pairs, score 82.75%.
Paired bootstrap 95% is [0.780, 0.875]; the conservative paired Hoeffding
interval is [0.691690, 0.963310]. There are no voids or technical failures,
and the single-match confirmation gate passes under this fixed clock/profile.

A fresh replay from every deal seed verified all 14,201 moves, reveal/capture
identities and terminal results. All 14,201 recorded actual-clock move charges
also fit within their banks even after allowing +/-0.5ms rounding per move;
no clock margin is ambiguous from rounding (minimum conservative margin 13ms).
The replay library hash and raw-game hash are published with the audit.

This supports a single-set advantage over this pinned released AB engine under
5+0.05 and the stated independent referee. Set B is still running; no trained improvement over the teacher, multi-training-seed
robustness, other time-control advantage or all-engine coverage is concluded.
AB remains defeated=false. Evidence: results/20261026-abj-confirmation-a.json
and results/20261026-abj-confirmation-a-audit.json.

The unchanged teacher completed the same 100 paired layouts at 149W/4D/47L
(75.5%; paired bootstrap 95% [0.6975, 0.81]), with no technical failures.
Replay verified all 14,251 moves and rounded actual-clock charges, with zero
ambiguous margins (minimum conservative margin 21ms). Thus much of the
advantage over this opponent is already present in the upstream backbone.
The learner scores 7.25 percentage points higher on this set, but its direct
teacher gate failed and this single opponent-specific contrast is not evidence
of a general learned improvement. Both set-B arms remain fixed and must finish.
Evidence: results/20261026-abj-teacher-a.json and
results/20261026-abj-teacher-a-audit.json.

## Second held-out learner set completed

Set B completed 150W/2D/48L in 200 games / 100 pairs, 75.5%, paired bootstrap
95% [0.6975, 0.8125], conservative Hoeffding [0.619190, 0.890810]. All 13,567
moves, reveal/capture identities, terminal outcomes and rounded actual clocks
passed replay; minimum conservative clock margin 6ms with no rounding ambiguity.
No technical failures or voids occurred. Both frozen learner sets now pass their
specified version/profile gates. The final unchanged-teacher set is still
running; the entire four-arm experiment is incomplete, and no general learned
improvement is established. Keep defeated=false until campaign audit concludes.
Evidence: results/20261026-abj-confirmation-b.json and its replay audit.
