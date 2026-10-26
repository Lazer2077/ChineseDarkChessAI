# Root sequential-halving ablation with observer-consistent search

Three low-simulation neural RL updates regressed, while reducing pending leaf
batch size improved fixed-clock warm-model play. A further hypothesis is that
root simulation allocation, rather than absent legal-action coverage alone,
limits useful improvement from the weak CNN critic. This experiment changes
root allocation while keeping original interior information-set PUCT.

The root mechanism is adapted from Danihelka et al.,
[Policy Improvement by Planning with Gumbel, ICLR2022](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf).
It samples a candidate set using log prior plus Gumbel noise, allocates equal
visits within a round, and rejects half after completed feedback. Ranking uses
log prior+noise+(50+max visits)*normalized root Q. Zero noise is included as an
evaluation ablation. The paper also specifies improved policy targets and
interior selection mechanisms; those are not implemented here. Its improvement
argument does not establish improvement with our approximate neural/particle
Q values or against adversarial hidden-information strategies.

## Defined implementation scope

`research/sequential_halving.py` provides a **fixed-work, root-only** research
variant. It explicitly requires a Gumbel seed, rejects time deadlines, Dirichlet
mixing and visit-temperature sampling, and returns a root-score recommendation
plus the actual normalized root visits. Those visits are not a completed-Q
training target. There is no new self-play training. A separate fixed-work actor
and its queued development matches are described below; no anytime profile is
claimed.

At256simulations and16candidates, rounds allocate64simulations each to16,8,4,2
actions. Other budgets use explicit rounding and may finish an incomplete
halving. Final selection uses the most visited set and root score, avoiding
insertion-order ties. Child values are converted to the root perspective and
normalized from[-1,1]to[0,1]. Independent Gumbel RNG does not consume the search's
world-sampling RNG; a common world seed still does not imply identical future
draws after methods choose different paths.

The original search body is copied in process with only two selection/batch
sites changed; originals and active experiments are untouched. Leaf batches
cannot cross a halving boundary. A boundary checks that all root visits are
backed up before using Q, so temporary virtual-loss credits never decide an
elimination. Chance navigation, synthetic actor-private observations, rule
history and backup signs remain original. Eight tests cover balanced/exact
work at large, odd and tiny budgets, correct root-Q perspective, pending-credit
rejection, real legal normalized particle search, actual opponent-slot poisoning
and independent reproducible Gumbel randomness.

## First three-root diagnostic

`results/root-halving-diagnostic-plan.json` fixes three existing TRAIN replay
roots (opening, last-own-dark, fully public), original warm checkpoint,
256simulations,LeafBatch8,oneCPUthread and noGPU. Each root has one PUCT search,
one zero-noise halving search and four noisy halving seeds,18searches/4,608actual
backups. These are correlated training roots, not held-out games or training-
seed replications. Histories are replayed before search.

An unchanged native teacher gives depth4forced-action scores for every legal
root action from the actor's observation FEN. Its search resets rule history;
these scores are an approximate teacher proxy, not exact Q, tablebase outcomes
or unbiased real-game returns. Proxy regret means best forced-action score
minus the method's selected-action score.

| Root | PUCT proxy regret | Zero-noise halving | Four noisy-halving regrets |
| --- | --- | --- | --- |
| Opening | 0cp | 0cp | 2,0,0,0cp |
| Last own dark | 178cp | 178cp | 178,291,252,178cp |
| Fully public | 628cp | 236cp | 236,236,236,236cp |

All18searches have exact visit totals and legal actions. The fully public case
changes froma8a7toa0a5; the partial-information cases show no zero-noise proxy
gain, and noise sometimes worsens the proxy. CPU elapsed times are about
1.56–1.75seconds/search on a shared machine; this is not an isolated GPU speed
comparison. Three roots cannot establish either strength or an observation-
phase interaction. Full scores, visits, rounds and timing:
`results/20261026-root-halving-diagnostic.json`.

A timed algorithm would require a separately defined anytime allocation and
recommendation rule; the fixed-work prototype must not silently treat a
million-simulation timed call as a completed halving.

## Fixed-work full-game contrasts

`research/fixed_work_profile.py` and the research-only UCI wrapper now expose
original PUCT, all-phase root halving, and fully-public-only root halving. The
last variant uses original PUCT until **both colors have zero on-board dark
pieces**, then switches root allocation. Private certainty does not trigger
the switch. Before that public transition, a targeted test reproduces original
PUCT outputs/visits/value sums exactly with the same seed. A real replay fixture
checks the fully public branch with actual rule history. The same CNN policy,
value function, chance model and interior PUCT apply to all profiles.

The wrapper explicitly converts each go request into a declared finite node
budget. It preserves the original go arguments as evidence; external actual
wall time is still charged by the referee. This is equal completed simulation
work, not equal FLOPs, neural evaluations or an anytime clock-controlled method.
Each fresh process records every root-edge visit/value sum, returned move,
public board/actor/history-coordinate key, method/phase and completed count.
An independent witness audit joins these to every actual timed go request and
every public native replay move. It verifies exact work and the public phase
gate alongside the separate native seeded-identity/legal/terminal/clock audit.

A two-game CPU integration smoke uses8simulations,LeafBatch8 and generous
actual300+2clocks, with the real frozen warm CNN. It completed2W/0D/0L but one
layout has no statistical strength meaning. All131moves, four fresh processes,
four warmups and actual simulation witnesses match; no fully-public positions
occurred, so this smoke does not exercise the phase switch. The separate real
full-root fixture covers that branch. Five corruption checks reject wrong
budgets, duplicate calls, false public flags, wrong boards and wrong selected
actions. The final audit binds the board and actor themselves to native replay,
in addition to the position key. See
`results/20261026-root-halving-cpu-smoke-provenance.json` and the referenced
native/work-v2 audits, raw games and compressed witnesses. This is integration
evidence, not improvement evidence.

`results/root-halving-matches-plan.json` fixes two fresh64-game development
contrasts before either outcome:

| Candidate | Control | Paired layout seed |
| --- | --- | --- |
| All-phase root halving | Original PUCT | 9460001 |
| Fully-public-only root halving | Original PUCT | 9470001 |

Both use the identical warm network,256completed simulations,LeafBatch8,
zero Gumbel noise, one CPU thread each, one shared dedicated GPU and two games
concurrent. Actual60+0.5clocks are identical generous fail-safes, defining a
different stratum from10+0.1teacher studies. Both fixed contrasts complete
regardless of results, with source/model checks and both audits. Sources also
pin the source-root diagnostic used to motivate the phase ablation. No tuning
or early stopping is allowed between these screens. If the public candidate
never reaches a fully revealed turn, that mechanism is unexercised rather than
positive or negative evidence for the switch. Every exercised phase count is
reported by the witness audit.

The study is queued behind the teacher-leaf campaign to use its released GPU;
it is not running concurrently on that GPU. No game-strength conclusion or
promotion follows from the prototype or the smoke. Future advantages would
initially be search changes of the unchanged warm network, not newly learned
weights, full Gumbel AlphaZero, or superiority over the native teacher.
