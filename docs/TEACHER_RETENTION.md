# Teacher-policy retention after neural self-play updates

The first function-level factorial contrast loses23W/0D/41L with the updated
policy and unchanged warm value, using matched two-forward inference. Increasing
training search effort from256to2048simulations also fails to improve the
warm-control screen. These results motivate checking policy supervision and
knowledge retention rather than assuming a critic-only bottleneck.

[Kickstarting Deep Reinforcement Learning](https://arxiv.org/pdf/1803.03835)
adds teacher-policy cross entropy to the RL objective on student-generated
trajectories, with an influence that can change during learning. It is useful
precedent for retaining a competent policy while optimizing reward, rather than
discarding teacher guidance immediately. The paper uses policy-gradient RL and
does not establish guarantees for our search-visit learner or Jieqi. Applying
its general distillation idea would be an attributed adaptation, not a new
algorithm by itself; a constant anchor would not reproduce its adaptive scheme.

Before changing training, `results/teacher-retention-diagnostic-plan.json`
freezes four already selected checkpoints: original warm, first256simulation
LeafBatch32RL,256simulation LeafBatch8RL, and2048simulation LeafBatch32RL.
Each is evaluated in FP32 on the same37,471positions from512original teacher
selection games. These are **reused checkpoint-selection data**, not newly
held-out test games. Warm checkpoint selection itself may favor the warm model
on this dataset. No model is fitted, selected or promoted by this diagnostic.

Report teacher-action CE/top1/top5, full2550-action KL from warm to each model,
and value error against original70%static+30%terminal targets, terminal-only
targets and static-only targets. Separate partial and fully public positions.
The full action distributions are not legal-masked policies; do not interpret
their entropy/KL as legal-action-only search divergence. Runs use one CPU thread
and no GPU, preserving all current actual-clock match GPU allocations.

Retention loss alone does not prove that imitating the old teacher is optimal,
or that teacher anchoring will improve play. A later intervention needs a fixed
control, compatible information inputs, new isolated match seeds and the native
teacher attribution control. Search-visit fitting and actor strength must still
be evaluated separately. No new training campaign is selected before the
diagnostic and remaining function-level contrasts finish.
