# Strong and research baselines: availability is part of coverage

The original jqe+it014 teacher remains the most immediate strength gate for our
learning changes. Released AB-JChess with its matching official network is a
useful independent comparator. Accepted older/classical variants are retained
as measured versions; they cannot substitute for every current neural engine.
The original confirmation criterion is above 50% score, not the user's new
stable-above 70% **wins** requirement. No current result satisfies the full goal.

## Additional availability audit, 2026-10-26

| Source | Audited evidence | Consequence |
| --- | --- | --- |
| [lichlzh/pikafish-jieqi](https://github.com/lichlzh/pikafish-jieqi/tree/c64d99705b8937bc38e21d148bd39441204e2988) | 45-entry tree and README: Qt/UCI client delegates AI to external Pikafish branches; no bundled neural weight or releases found | GUI, deduplicate against its actual engine; not another independent AI |
| [swim2sun/Pikafish-jieqi2022](https://github.com/swim2sun/Pikafish-jieqi2022) | Additional pinned jieqi/ablation branch trees contain no required network, release list empty; default remains `pikafish.nnue` | Exact compatible trained weight remains unresolved; do not replace it with an arbitrary Xiangqi network |
| [zball-bz/Jieqi-hybrid](https://github.com/zball-bz/Jieqi-hybrid/tree/a556210806706c0aab554fc482b52cacf3b38593) | 90-entry tree includes C++ PIMCTS and neural self-play, but the loader's `eval_small.safetensors` is absent; supplied JSON files are architecture configurations | Distinct neural candidate stays unverified; random initialization would not measure its intended strength |

The latter project also needs a component-specific license audit: inherited
LCZero C++ files retain GPL headers with CUDA permission; `game/LICENSE` contains
pybind's BSD license. A missing root license does not erase inherited headers,
and that vendored notice does not establish BSD terms for the entire new project.
New Python/search components remain unresolved. We have not reused that code.

The GUI author's reports of unsuccessful historical network downloads are useful
leads, not independent reproductions by this project. Our findings are limited
to the specified trees, files and release endpoints. We do not claim to have
exhausted all historical commits, community archives or external model hosts.
Source hashes, branch revisions and findings:
`results/20261026-neural-baseline-availability.json`.

## Research papers versus runnable opponents

[Wang and Lu, On the Complexity of Dark Chinese Chess, 2021](https://arxiv.org/abs/2112.02989)
studies full-board Jieqi with captor-private dark captures. It estimates game and
information-set complexity using random-move self-play, with a 40-ply no-capture
termination rule. That experiment is not a supplied trained strong-playing
baseline, and its rules do not match our 80-ply benchmark profile.

The paper also explicitly excludes off-board hidden status when the corresponding
color has no on-board dark piece. This is relevant prior work: forgetting
future-irrelevant inventory is not a new fundamental observation of this project.
Our phase representation, value heads and graph-sharing code require their own
controlled measurements and accurate attribution. Complexity estimates alone
cannot calibrate playing strength or establish that all research baselines have
been defeated.

The report gate requires both current strong runnable opponents and an honest
account of missing/incompatible candidates. Inventory size is not a strength
rating, and many forks or GUI clients do not imply algorithmic diversity.
