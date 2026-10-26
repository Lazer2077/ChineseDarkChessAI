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
