# Experimental weights — not promoted

warm-20261026.jqn is the small warm-start research candidate, SHA-256
114a1dd460b66b52cd8089bf6f5fbd2273ed90fc4aa9f7d8ef35ea068cecae1f.
It is derived from jqe's MIT-licensed it014.jqn, not a model trained from scratch.
The original model is at devinokeefe/jieqi commit
331a6e612a038c4a04cf980f7b9482910fd50078, engine/nets/it014.jqn.
Original authorship/licence is retained in LICENSE.jqe.

Fine-tuning recipe: 512 teacher selfplay games, 5,882 held-out validation rows,
lr 1e-5, lambda 0.9, K 430, 10 epochs, batch 1024, seed 20261026. The data and
training method are documented in docs/ADAPTIVE_SEARCH.md. Development score
18-1-13 did not establish superiority. Confirmation A finished 95-10-95 and
included one time forfeit; it does not pass promotion. Confirmation B finished 77-17-106 (42.75%) with one time forfeit. Neither
confirmation supports promotion; do not pool them with development results.

Do not advertise this checkpoint as the best Jieqi AI or as an improved upstream
release. It is published so completed research matches can be independently run.

## Expanded first-iteration candidate — not promoted

expanded-seed20261026-it1.jqn, SHA-256
9cd00a5f4fb7ccceb5c1374660a623acb60cd18eafa79f4c6aa9848b0e278833,
is derived from the same upstream MIT it014.jqn; LICENSE.jqe applies to its
upstream origin. It was trained on 8,192 completed selfplay games (405,959
retained positions), 15 epochs / 1,485 updates, batch 4096, lr 2e-5, lambda .7,
K 430, learner seed 20261027. Selection uses 25,031 separate validation positions
and chose epoch 1. The generation and model hashes are documented in the
expanded-series results. This is a learned value model on an inherited search
architecture, not a new search engine trained from scratch.

Its 200-game gate versus its teacher finished 87W/13D/100L, so it was not
accepted as the next actor. A separate AB-JChess v0.2d development screen at
actual clock 5+0.05 finished 27W/1D/4L; 16 pairs do not establish confirmed
superiority or a training gain. Frozen confirmation and unchanged-teacher
control matches are separate. It is published for reproducibility, not as a
confirmed improvement over the teacher.
