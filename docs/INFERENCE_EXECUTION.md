# Frozen neural arithmetic and actual-clock search work

The learned policy/value branch remains far below the unchanged native teacher.
This study examines inference cost before committing more computation to larger
models. It is an execution experiment, not a new POMDP algorithm, new network
training or a demonstrated playing-strength improvement.

## Per-call BF16 does not help this small-batch path

Following the matching installed-version
[PyTorch AMP documentation](https://docs.pytorch.org/docs/2.7/amp.html), keep
stored parameters FP32 and enter BF16 autocast separately for each forward.
Softmax and returned values explicitly use FP32. Use128existing TRAIN records,
the first32in each live-dark bin D0/1–4/5–14/15–30, interleaved; no held-out match
or checkpoint selection uses these records. Batch1contains a D0row; batch8/32
cover all four bins. Three alternating-order repeats exclude five warmups and
time20calls each. One CPU thread and a separate inference device are used.

At batch32, complete encoding/forward/CPU-return calls take8.63vs11.02ms for
the trained64x15warm network,8.73vs10.88ms for the earlier trained128x15dense
network, and9.08vs10.97ms for a cold128x15relative network (FP32vsBF16 medians).
BF16 is about21–28%slower here. Trained-model mean policy total variation is
about0.0106–0.0109; value maximum absolute differences are0.0105–0.0125. These
are full-geometric-action probabilities, without actual legal masking, and not
played-game disagreements. The cold relative model is an architecture/runtime
control; its numerical behavior does not validate future trained weights.
No BF16 actor or match follows this negative cost result.

Evidence: `results/inference-precision-cost-plan.json` and
`results/20261027-inference-precision-cost.json`.

## Capture the same forward operations instead

[PyTorch CUDA graphs](https://docs.pytorch.org/docs/2.7/notes/cuda.html#cuda-graphs)
reduce repeated host dispatch for static tensor shapes. `GraphPolicyValue`
captures only the model forward, retaining original FP32 inputs/weights and
unchanged TF32 backend settings. Side-stream warmups precede capture. Static
input buffers are fully overwritten on each call, returned outputs are cloned
to preserve their lifetime across later calls, and uncaptured shapes use the
original eager network. The frozen wrapper forbids training and use from a
different inference thread. It is for a fixed model, not an arbitrarily mutable
module or a shared multi-threaded inference server.

Observer encoding, private information projection, full rule histories, legal
masks, information-set keys, random chance sampling, PUCT allocation/backup and
deadline accounting all remain eager and unchanged. No actual dark identities
are added. This does not merge beliefs, assume translation symmetry or change
the model's existing history-omitting input contract.

The same128TRAIN inputs at three batch sizes give bitwise-identical policy/value
outputs for both trained dense models and the cold relative model. Replays
preserve earlier returned tensors, noncaptured-shape outputs match eager, and
all weights/normalization buffers remain unchanged. At batch32, eager versus
captured complete-evaluation medians are8.95/3.25ms (warm64),9.04/4.23ms
(trained dense128) and9.24/4.18ms (cold relative128):2.75x/2.14x/2.21x microbenchmark
ratios. Capture/setup is excluded. These are repeated correlated inputs, not
end-to-end search ratios or a universal numerical-equivalence proof.

Evidence: `results/cuda-graph-inference-cost-plan.json` and
`results/20261027-cuda-graph-inference-cost.json`.

## Complete-search controls before timed matches

Reuse the three previously declared TRAIN roots: opening, uncertain last own
dark and fully public. Two trained dense backbones each run256completed
simulations in eager/captured modes, with three alternating-order repetitions,
identical per-root RNG, no Dirichlet noise and unchanged full observer/rule
histories. All1..32batch shapes are preloaded outside timing. All18paired root
fingerprints agree exactly: move, complete2550-action visit policy, root visits
and value sum, and all child visits/value sums/priors. Total measured work is
9216simulations; these are three correlated roots, not independent games.

| Trained model | Opening eager/graph | Last-own-dark | Fully public |
| --- | ---: | ---: | ---: |
| Warm64x15 | 1.317 | 1.412 | 1.267 |
| Earlier dense128x15 | 1.663 | 1.783 | 1.222 |

Thus measured warm-model search time falls about21–29%; wider-model time falls
about18–44%on these roots. Incomplete observer encoding and other search costs
remain, so inference microbenchmark ratios do not transfer directly. This is
bounded fixed-work equivalence/runtime evidence, not stronger decisions at the
same work or success against a baseline. Source/input hashes and full root
fingerprints: `results/cuda-graph-search-control-plan.json` and
`results/20261027-cuda-graph-search-control.json`.

The prespecified follow-up condition—all fingerprints equal and every warm64
per-root median speed ratio at least1.10—is met. A separate frozen actual-clock
screen compares the same warm weights with graph versus eager execution at
10+0.1/LeafBatch32,64games, seed9680001; another64games at9681001retain the
unchanged native teacher. Both neural sides use the same safe model loader.
Graph shapes preload at `isready` before move clocks, with no shape capture in
timed search; both contrasts finish regardless of the first result. Require raw
archives and independent native/clock audits, retaining technical/ambiguous
flags. Recipe: `results/cuda-graph-actual-clock-evaluation-plan.json`.

The already frozen wider dense/relative12-epoch study and all five of its eager
evaluation contrasts remain unchanged. Graph execution is not silently added
to those actors or used to select their checkpoints. No public push or final
>70%-win/report claim follows from runtime evidence alone.

## First actual-clock screen: no demonstrated playing-strength gain

The graph-versus-eager warm-model screen completes30W2D32L/64,
48.4375%score, paired bootstrap95%[0.3828125,0.59375]. All3333moves pass the
independent native reveal/legal/terminal/clock audit, with zero technical games,
voids or rounding ambiguities. Thus the bounded fixed-work speed improvement
does not establish a playing-strength improvement at this clock. This sample
also cannot rule out smaller effects. Existing records lack completed per-move
simulation counts, so similar mean move times alone cannot explain the result.
The unchanged-teacher contrast continues without retuning. No promotion or
larger confirmation campaign is justified by this first screen. Fixed-work
generation efficiency remains a separate possible use, requiring equivalence
checks for any newly trained checkpoint before adopting it.
Evidence: `results/20261027-cuda-graph-vs-eager{,-audit}.json`.
