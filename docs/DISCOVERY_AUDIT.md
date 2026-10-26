# Discovery audit — 2026-10-26

The inventory is an ongoing research checklist, not a certificate of exhaustive
coverage. A new search found missing candidate engines, so the previous small
inventory must not be treated as the entire open-source Jieqi field.

Queries and raw repository leads are retained in discovery-20261026.json.
Connected GitHub search used `jieqi chess`, `揭棋`, `jieqi AI`, `JieqiAI`, and
`dark chess xiangqi`, plus Chinese AI and Vietnamese name variants, page 1 with
at most 100 results per query. Additional raw leads are retained in
discovery-20261026-extended.json. The Chinese
query reaches that cap and includes irrelevant book repositories. Search
results are leads, not verified independent engines. Default HEADs are resolved
and pinned separately in opponents.json. Forks are not deduplicated from names,
repository sizes, or README similarity alone.

## Sources inspected locally

| Source | Verified evidence | Remaining coverage boundary |
| --- | --- | --- |
| Yuan-Bai/jieqi-lab, 4173aa3 | MIT; independent Python information-set search and neural selfplay code; 32-simulation opening probe returns a legal move | No pretrained neural weights distributed. Current `ris_ismcts` explicitly describes root-consistent sampling and actor-observation sharing, retaining an older RIS function name; do not equate it with full re-determinization at every decision. Rule-v5 six-round check/chase bans and repetition differ from the present native referee; fair profile/adapter still needed. |
| EtheXReal/jieqi-ai, 6009bcf | Independent Python alpha-beta code plus neural research code found | No root license found. README says neither player learns hidden-capture identities; the UCI replay function reads only move coordinates and does not consume identity suffixes. A standard private-capture benchmark cannot silently treat this as a compatible full-history adapter. |
| zball-bz/Jieqi-hybrid, a556210 | C++ game/search and Python selfplay/training code found | No root license; game/LICENSE is a bundled Pybind Development Team notice, not sufficient evidence that all new engine/training code is licensed under it. Rule/model/API audit pending. |
| YOLOW92/moyi-jieqi, 5b82525 | GPL-3.0; independent full-board game/AI source | Cross-colour shuffle: a reveal can transfer the piece to the opponent. This is not the standard separate-colour Jieqi game and is incompatible with current training/evaluation profiles. |

Public opening-probe evidence is in results/20261026-yuan-lab-source-audit.json.
It proves the pure-search entry point runs, not a completed match or strength.
Neural-weight unavailability must not be used to omit its runnable pure-search
variant from future adapter work.

## Leads retained for further inspection

The expanded inventory also pins the Java candidates Makinuohara/2026-jieqi-AI,
zhzjzz/AI-Jieqi, Lazyy-star/AI-Jieqi and zhiduoming/Unveil; the standalone C++
JSON engine wangyiming1628/jieqi_ai; and candidate variants from rexfang0829,
iwestlin, calm232, igofreely and dudu555-djiao. These entries remain unverified.
The wangyiming project includes screen/mouse automation, but only its standalone
engine is relevant: no external game interaction is run or authorized here.
A missing README.md is not proof that a repository or engine is unavailable.

Wu200008/wilderness-xiangqi's README describes cross-colour mixing and public
hidden captures, so it is separately flagged as a rule variant. Neither that
variant nor moyi is claimed defeated. Possible Banqi/game-only leads require
source verification before inclusion as compatible independent engines.

No newly discovered entry is marked defeated. No license is inferred from
public visibility, an ancestor, a dependency notice, or a repository's name.

## Pika archive and hybrid follow-up

The connected GitHub searches `PikaJieQi`, `pikafish jieqi fork:true`, and
`"揭棋" "NNUE"` found further leads; the latter returned none and is not proof
that no such engine exists. Results and provenance are preserved in
`discovery-20261026-pika-archives.json`. Search matches are not automatically
independent engines or compatible full-board games.

Three unchanged default checkouts were examined. `swim2sun/Pikafish-jieqi2022`
contains real hidden-piece chance-search code, with GPL source and a distinct
restrictive NNUE license; its required historical `pikafish.nnue` is not bundled
and an exact weight/version remains unresolved. `thanh25896/PikafishJieqi` has
ordinary Xiangqi types and no hidden-piece implementation found in its default
`src`; other branches/history remain unaudited. Repository names alone cannot
establish compatibility. `hefengfan0615/pikafish-jieqi` contains a distinct
Pika-search/ABJ-V11 NNUE hybrid; its own network download script explicitly
selects the released AB-JChess v0.2d network. It is not silently deduplicated with
AB-JChess or classical Pika.

The hybrid was built with a declaration-only GCC11 compatibility repair, using
the identical already verified official network it requests. Two actual-clock
smoke games completed, and all 57 moves/rounded clock charges replayed cleanly.
A separate 32-game development comparison is running; neither build nor smoke
is strength evidence. Source, license, model and binary hashes are in
`results/20261026-pika-archive-source-audit.json`; reproduction is
`scripts/build_hefeng_pika.py` after obtaining the pinned checkout and official
network through the existing source/asset workflows.

The already validated Universe observer adapter passed a new actual-clock smoke
with 35 replayed moves. Two frozen 200-game sets are running with original hard
search, complete observer-specific histories, 5+0.05 clocks, separate layout
seeds and no early stopping. Universe ignores Threads/Hash settings and retains
its own data structures; no equal hash-memory claim is made. Its internal
repetition policy differs from the external native referee, as documented in
the original adapter. Recipe: `results/universe-clock-confirmation-plan.json`.
Planned sets and partial scores never mark an opponent defeated.

## Completed confirmations and new information-set adapter

AB-JChess v0.2d and the pinned Universe hard-search profile now pass both fixed
confirmation sets, including full move/clock replay and post-campaign frozen
hashes. Acceptance is version/profile-specific; details and teacher attribution
are in [CONFIRMED_RESULTS.md](CONFIRMED_RESULTS.md). No general learned gain
or all-engine coverage is inferred. The classical Pika original fails its
two-set clean-run criterion because B includes an opponent time forfeit.

The independent Yuan-Bai root-consistent ISMCTS now has an observer-only clock
adapter with original optional native acceleration. Both backends passed
960-position checks, and native-backed development completed32/0/0. Frozen
pure-search confirmations are running. No pretrained weights ship upstream,
so its neural mode remains unavailable/unverified. Interface, algorithm and
rule-profile limits are in [YUAN_NATIVE.md](YUAN_NATIVE.md).

The Pika/ABJ-NNUE hybrid has multiple process-exit voids in its frozen campaign.
Fresh-process replays of three failed observation prefixes, including recovered
original clock commands, did not reproduce the exits. Earlier search/transposition
history and host scheduling were not reproduced. Cause remains unresolved; all
voids continue to invalidate clean confirmation. Diagnostic evidence:
results/20261026-hefeng-fault-replay.json. No opponent source is patched to
turn this into a pass.

Pinned archive catalog IDs match their checkout directories: `pika-hefeng-archive`,
`pika-jieqi2022-archive`, and `pika-thanh-archive`. They can be verified/fetched
through `scripts/fetch_sources.py` without overwriting existing sources. Their
GitHub full names remain explicit metadata; no lineage deduplication is implied.
