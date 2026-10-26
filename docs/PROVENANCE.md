# Sources and licence
Vendored engine.js, ai.js and LICENSE from https://github.com/arthuryangcs/mist-jieqi
(original requested URL https://github.com/arthuryangcs/JieqiAI redirects there).
Pinned commit: da1ab8ce11659110f19422213c5811cbfab77cb6. Retrieved 2026-10-26.
Files are unmodified; GPL-3.0 applies to this project. Preserve upstream notices.
The baseline is the pinned JavaScript search, not the upstream C++ engine; distinguish these in reports.

Research references:
- Silver & Veness, Monte-Carlo Planning in Large POMDPs (2010): https://papers.nips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf
- Brown et al., Combining Deep Reinforcement Learning and Search for Imperfect-Information Games (2020): https://arxiv.org/abs/2007.13544
No code from these papers is included.

## Native teacher feasibility experiment
`scripts/native_pilot.py` uses the external MIT-licensed jqe engine and trainer at
331a6e612a038c4a04cf980f7b9482910fd50078 unchanged, including its published it014.jqn
as teacher. It writes newly trained weights to the ignored runs directory.
This tests reuse of a strong native search/NNUE foundation, as requested by the
project, and is explicitly distinct from our root-sampling implementation.
The unmodified backbone and teacher are not claimed as original contributions.
No outcome against the teacher is attributed to our algorithm until substantive
changes and independently evaluated newly trained weights exist.

## Native research backbone and AB-JChess opponent
Our native experimental allocation is applied by scripts/build_native.py to an
unchanged fixed MIT jqe source checkout; the source copy retains LICENSE.jqe.
The independent referee and adapters are from that same MIT project. The
protocol entry wrapper only chooses strict versus classical reveal padding.
AB-JChess v0.2d source is fe35d87c7553fd23e8aab3a1dde7e712f286b416 (GPL-3.0).
Its official release asset and network hashes are in docs/opponents.json. We
use those weights locally and do not redistribute them. scripts/build_abj.py
records a GCC11 declaration-only compatibility repair; search and evaluation
function bodies remain unchanged. No Windows executable from the asset is run.
