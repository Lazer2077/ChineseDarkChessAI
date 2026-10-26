# justLxy/Jieqi: active engine and interface audit

Pinned source: https://github.com/justLxy/Jieqi/tree/9d826d7ff07efea3871417233c2f730997862d13

The active web backend imports `cppjieqi`, creates persistent game instances,
and calls `set_board` followed by `get_ai_move`. Its search is C++ `AIBoard5`,
not merely one of the repository's Python musesfish copies. The engine remains
an independent variant in the inventory; shared ancestry does not justify
silently deduplicating it.

## Completed evidence

The original CMake project and bundled pybind11 build a Python 3.11 extension
with GCC 11.4 and CMake 4.4.3, without changing bindings or search. The original
`initialize`, `create_game`, `set_board`, `get_ai_move`, and `delete_game` calls
complete an opening probe, returning legal `i3i4`. The module must be initialized
with a working directory where the original `../score.conf` and `../kaijuku`
resolve; otherwise it can run with missing evaluation resources, which would
not be a faithful baseline. No Flask server is required for this probe.

Build and probe checksums are in results/20261026-justlxy-module-audit.json.
This is build/interface evidence, not a strength result or complete game.
The original LICENSE.md contains the GPL v3 text with a nonstandard header
date; that irregularity is recorded rather than silently corrected.

## Outstanding information and time-control issues

`cppjieqi/bindings.cpp` exposes a 90-character public board, side to move, and
public-board history. It has no argument for the acting player's private
captured identities or residual inventory. `set_board` constructs a temporary
`Board`, calls `GenerateRandomMap` and `initialize_di`, and copies those generated
inventories into the thinker. In `board/board.cpp`, `GenerateRandomMap` removes
visible on-board identities from the initial multiset, then samples identities
for occupied hidden starting squares. This is not the same as conditioning on
all public reveals plus the actor's private captured identities.

Passing true hidden identities to fill this gap would leak information. Passing
only the public board faithfully reproduces the web interface but does not
establish compatibility with the project's private-capture observation profile.
An explicit observer-interface extension or separately justified public-only
profile is needed before a fair match; search changes must remain separate.

`AIBoard5::Think(int maxdepth)` explicitly ignores its argument. The active
`mtd_thinker5` searches depths 5 through 7 with a quiescence extension and tests
an internal 15-second elapsed threshold only after a completed iteration.
The exported API offers no stop command or user move-time budget. A Python
argument labeled depth therefore cannot be used to assert equal computation.
Forcibly terminating an unfinished search and counting it as a candidate win
would not establish a strength advantage.

## Reproduce the validated build

```sh
python3 scripts/fetch_sources.py justlxy-jieqi
python3.11 scripts/build_justlxy.py
```

The build recipe writes local provenance under build/justlxy-module; it does
not modify or redistribute upstream source. The published probe metadata
contains no machine identifiers or local environment paths.
