# Original reference C++ baseline

The original requested arthuryangcs/JieqiAI URL redirects to mist-jieqi. Its
pinned da1ab8ce11659110f19422213c5811cbfab77cb6 includes a C++ search distinct
from its JavaScript fallback. Previous JS results do not cover this C++ engine.

## Baseline transport and information
adapters/mist_native.mjs imports the original unmodified native/bridge.mjs and
uses its original C++ binary, 8 balanced hidden-world samples, depth cap 8, and
observationSeed. It declares a four-thread allowance; it cannot silently be
called a single-thread baseline. No engine search/score/source change is made.

src/history.js reconstructs the actor's observation from public reveals and
only that actor's private captured identities. It rejects opponent-private
capture identities and missing reveals. Random-game tests compare the result
with an independent true-world simulator at every queried position. Hidden
board identities are absent from the final observation passed to the bridge.

## Separate rule profile
The standard native jqe referee uses an 80-ply capture clock and check/chase
adjudication. Mist uses a 120-ply capture-or-reveal clock, third repeated public
position last-mover loss, 400-ply draw, mate/stalemate precedence. Mixing these
without reporting the profile would not establish a fair original-rule match.

scripts/mist_profile_match.mjs uses the pinned Mist game rules as authority.
The candidate's copied native source is adapted by build_mist_profile.py:
120-ply threshold, reveal resets, third repetition last-mover loss, total-ply
cutoff. Its search is based on the attributed jqe backbone; the opponent remains
unmodified. Native diagnostics check reveal reset, second occurrence ongoing,
and third occurrence last-mover loss. Search null-move/cycle approximations
remain heuristics, not independent formal referee implementations.

Both engines receive complete actor-specific histories, four-thread allowances
and the same movetime. Networks are preloaded before games. Actual elapsed
search time is recorded. Budget + 1000 ms overrun quarantines a screen game,
never gives the candidate an unlogged technical win. This tolerance is for an
integration screen; later formal clocks must account for all time explicitly.

The initial development screen is two paired games, 1000 ms/move, seed 7400001,
using the published unpromoted warm checkpoint, not the unchanged teacher.
Small samples are not superiority evidence. Original-rule results are reported
separately from jqe-rule results; they are not pooled.

## Reproduce
```sh
python3 scripts/fetch_sources.py mist-jieqi devinokeefe-jieqi
CXX=g++ node external/mist-jieqi/native/build.mjs
python3 scripts/build_mist_profile.py
MIST_NODE=node node scripts/mist_profile_match.mjs --games=2 --movetime=1000 --seed=7400001 --net=models/experimental/warm-20261026.jqn --out=runs/your-new-mist-screen
```
Original baseline is GPL-3.0. The copied candidate backbone is MIT and its licence
is preserved in the build. No inference result from this screen certifies all
rule variants, all clocks, or the continuous-learning models still in training.

## Completed smoke and expanded screen
The own-rule smoke completed two games with two candidate wins by mate, 37 and
34 plies, no voids. Compact evidence and checksummed raw games are published in
results/20261026-mist-native-smoke.json. The unpromoted warm checkpoint is now
available under models/experimental with upstream attribution and its negative
confirmation result. An expanded 16-game development screen uses a fresh seed
7400101, the same 1000-ms/four-thread profile, and does not pool with other rules.
