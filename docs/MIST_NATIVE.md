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

The expanded screen completed 15 candidate wins, no draws, and one loss, without
void games. Its eight layout pairs are insufficient for formal confirmation.
The paired bounded-mean 95% interval still includes 50%; the integration timing
tolerance also prevents treating this as a strict-clock tournament. Evidence:
results/20261026-mist-native-development.json and checksummed raw games. No
opponent is promoted to defeated.

## Actual-clock confirmation protocol
scripts/mist_clock_match.mjs removes the integration-screen grace period.
Both engines receive the same 15+0.1 second game bank, four-thread allowance,
and external per-move allocation: floor(min(60000, max(1, bank-20),
bank/20 + 0.8*increment)), at least 1 ms. This common external time manager is
part of the comparison condition; it is not a claim about each engine's best
native time-management policy. Full wall time from position submission to
bestmove is charged, including observer reconstruction and transport. Startup
and public-start warmup are excluded symmetrically. Increment is credited only
after an on-time legal move; it cannot rescue a flag fall. No extra time is
granted above the bank. All technical results remain in WDL and prevent gate
acceptance; infrastructure voids also prevent acceptance.

Two complete smoke games at 3+0.05 ended by mate with no technical errors.
Every move's actual bank charge and terminal result was replay-verified.
Frozen confirmation sets each contain 200 games / 100 layout pairs, seeds
7701001 and 7702001. The same unpromoted warm candidate is used throughout,
with no stopping on interim results or tuning between sets. Each set must pass
paired bootstrap and Hoeffding lower limits above 50%, with no technical
failures; one policy and one time manager cannot establish multi-training-seed
or all-engine superiority. Frozen source/artifact hashes are checked before
every game and at completion. See results/mist-clock-confirmation-plan.json.

## First actual-clock confirmation set completed

Set A completed 200 games / 100 pairs at 192W/0D/8L (96%; paired bootstrap
95% [0.93, 0.985]). Four wins were opponent time forfeits. Under the frozen
zero-technical-failure rule, valid_technical_run=false and the confirmation
gate fails despite the high score. The four failing Mist attempts occurred
after 121, 127, 127, and 136 completed plies; observed remaining banks were
about 118–166ms. This locates an interface/time-management issue for further
diagnosis; it does not establish its cause or justify discarding the games.
Set B continues unchanged. No clean 188-win subset or retrospectively relaxed
time allowance is presented as a confirmation. Evidence:
results/20261026-mist-clock-a.json.
