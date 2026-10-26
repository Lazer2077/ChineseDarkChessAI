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
