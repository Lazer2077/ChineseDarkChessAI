# Musesfish timed compatibility profile

The GPL-3.0 engine [miaosiSari/Jieqi](https://github.com/miaosiSari/Jieqi) is pinned
to `3de6c17ae6dafa035b5eea5ec7d6b7f39118aab6`; its original source remains unchanged.
An earlier 16-game development result was insufficient for confirmation.

The unchanged upstream jqe MIT shim and packaged worker already reconstruct the
entire observer history, including private dark captures. Their native-public-root
profile filters root candidates by external-referee legality, retains the
opponent's heuristic exclusions, and interrupts search with a Linux timer. Only
completed iterations are used; if none finishes, the existing highest-static-value
eligible root fallback is explicit. Real hidden identities are inaccessible to
the search. This is a disclosed compatibility profile, not an unchanged original
UI/search invocation or a characterization of every Musesfish configuration.

The worker searches depth 1 through at most 7, whereas the original engine's
`Searcher.search` iterates depths 5 through 7. Lower-depth completed fallbacks
permit finite game-clock use, but change the original invocation. Record search
depth/completion information in further diagnostics before treating this profile
as a strong baseline. External native rules govern termination; internal search
rule approximations can differ. No strength ranking follows from engine identity.

## Game-clock conversion

The original shim handles `go movetime` and otherwise defaults to 1000ms. The new
`adapters/musesfish_clock.py` converts an actual clock to an explicit budget:

```math
b_t=\max(1,\lfloor\min(T_t/25+0.75I_t,T_t/4,60000)\rfloor)\text{ ms}.
```

The original shim then reserves its existing 30ms before the timed worker.
The external clock charges all proxy, observer-replay and search time with zero
extra grace. The converter does not modify private suffixes or search code. It
counts the whole supplied history to select the correct side's bank; unsupported
positions or missing/nonfinite clocks are rejected rather than silently using
one second. Hash/Threads options remain ignored by the single-thread worker
with its fixed table; equal native Hash allocation is not claimed.

## Frozen campaign

`results/musesfish-clock-campaign-plan.json` pins the candidate, original sources,
shim, packaged worker, referee and new wrapper/controller. It fixes actual
10+0.1 clocks, one CPU thread each, two concurrent games, and these sets:

| Stage | Games | Color-swapped pairs | Layout seed |
| --- | --- | --- | --- |
| Smoke | 2 | 1 | 8510001 |
| Development | 32 | 16 | 8520001 |
| Confirmation A | 200 | 100 | 8530001 |
| Confirmation B | 200 | 100 | 8540001 |

Every stage is independently replayed for seeded identities, legality, terminal
outcomes and rounded actual clocks. Technical failures, voids, missing pairs or
ambiguous clock margins stop the campaign; development must score above 50%
before confirmation. Both fixed confirmations remain required. No model or
clock retuning occurs between sets. The candidate uses the experimental small
NNUE whose superiority over its own upstream teacher remains unconfirmed.

Smoke completed at 2W/0D/0L with all 115 moves and clocks cleanly replayed,
zero technical failures/voids and minimum conservative remaining clock 2482ms.
Evidence: `results/20261026-musesfish-clock-smoke.json` and its `-audit.json`.
The 32-game development set also completed at 32W/0D/0L; all 1,282 moves/clocks
replayed cleanly, with no technical games, voids or rounding ambiguities
(minimum conservative remaining clock 760.5ms). Development evidence uses the
same `results/20261026-musesfish-clock-development{,-audit}.json` naming.
The two confirmation sets remain pending; this development screen does not
replace them. Any later
acceptance is scoped to this exact profile. It neither establishes >70% wins
against all strong baselines nor removes other configurations from the inventory.
