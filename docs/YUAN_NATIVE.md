# Yuan-Bai root-consistent information-set search baseline

Pinned MIT source: `Yuan-Bai/jieqi-lab` at
`4173aa3ddc72e19227a619ffc4ea80a88b3ba2b4`.
The repository explicitly does not ship pretrained weights. We evaluate its
pure heuristic-rollout ISMCTS mode, not an invented untrained neural baseline
or a claim against an unavailable trained model.

The original `ris_ismcts` name is retained upstream, but its actual algorithm
samples a root-consistent world once per simulation and shares decision
statistics by the acting player's information key. It is not re-determinization
at every decision or an equilibrium solver. Original exploration, 60-ply
heuristic rollouts and material cutoff scale are retained.

## Observer bridge and private captures

`adapters/yuan_observer.py` receives only one player's strict UCI history. It
reconstructs public geometry, reveals, repetition/check/chase state and counters.
It separately tracks public type counts, private captured identities known to
this viewer, and unknown identities lost to the opponent. It rejects leaked
opponent capture identities and missing identities on the viewer's own captures.

The upstream rule transition requires a type for every hidden capture. During
public-history reconstruction we permit a temporary sampled type, resetting the
transition pool from public counts before each move. That temporary removal
never becomes a constraint on later public reveals or player inventory. At the
end, the real observer ledger conditions a new root particle. No true layout is
read by the adapter. The independent oracle uses seeded truth only for checking.
The original simulation belief builder reconstructs its hypothetical opponent
from root information and the sampled world, not an actual opponent-private
ledger supplied by the bridge.

Two 960-position checks, one Python and one with original Cython acceleration,
matched the independent native referee's boards, actor inventories, public
counts and no-capture counters. Each included 27 private captures, 96 checks
against different arbitrary hidden-past placeholders and 54 rejected private
suffix faults. Legal sets agreed in these samples. This does not prove complete
rule equivalence: Yuan's default tiantian v5 includes six-round check/chase
bans and third-repetition draw, while the external benchmark referee adjudicates
repetition/check/chase by its own profile. Rule history is never reset to make
these differences disappear.

## Timed search and original acceleration

`adapters/yuan_uci.py` imports the original search and inserts only a deadline
check **between simulations**. It does not change selection, information keys,
rollouts, backup, cutoff values or priors. At least one real simulation completes;
there is no random timeout fallback. Fixed-iteration tests with no deadline
match original actions and root statistics, including private uncertainty.
An already expired deadline matches precisely one original completed simulation.

The original optional rule and legal-move Cython extensions are built outside
the untouched checkout by `scripts/build_yuan_native.py`. Both are enabled for
timed comparisons; hashes and source fingerprints are checked. We do not
artificially benchmark only a slower fallback when its original acceleration is
available. Build metadata and both backend audits are published.

Each decision uses bank/25 + 0.75*increment, capped at bank/4, minus 20ms overhead.
The complete observer reconstruction and search wall time is charged to the
actual game bank. A long simulation can exceed its allocated move allowance;
this remains visible in timing and can lose on the game clock. Hash/Threads
options are ignored because pure search is single-threaded with its own data
structures. No equal hash-memory claim is made.

The native-backed two-game actual-clock smoke passed, with all 53 moves and
clock records replayed. A separate 32-game development screen at 5+0.05 finished
32/0/0, without technical errors, but has only 16 layout pairs. A frozen two-set
200-game confirmation campaign now uses new seeds with no early stop or retuning.
Only the pure-search adapter profile is tested; no neural-mode or all-engine
superiority is inferred. Recipe:
`results/yuan-native-confirmation-plan.json`.

## Reproduce the source and native build

Fetch the catalog ID `yuan-bai-jieqi-lab` with `scripts/fetch_sources.py`, then
run `scripts/build_yuan_native.py` with Python3.11 and Cython available. The build
uses the original tools and writes outside the source checkout. Set
`YUAN_NATIVE_DIR` to `build/yuan-native/source` and request the `cython` rules
and legal backends for the recorded native-backed profile. Missing native
modules are an error in that profile, not a silent fallback. Use a separate
process from the unrelated native referee package, which also imports as jieqi.

## Pure-search confirmations completed

Both frozen200-game sets completed200W/0D/0L, with no technical failures or
voids. All11,624 moves and actual-clock charges replayed cleanly; both separate
conservative Hoeffding95% lower bounds are86.42%. Post-campaign source/native
binary hashes still match. Evidence: results/20261026-yuan-pure-acceptance.json.
This accepts only the stated accelerated pure-search observer/clock profile.
Its unavailable pretrained neural mode remains unverified, and the project-wide
`defeated` flag is not set. Our separately trained policy/value derivatives are
new research candidates and do not stand in for an author-released model.
