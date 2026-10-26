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
