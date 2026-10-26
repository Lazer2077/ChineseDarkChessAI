# Frozen Pika-variant actual-clock confirmations

The original, enhanced, and Mistboard/WASM Pika variants remain separate
opponents. Their previous 16-game screens are development results, insufficient
for confirmation. The frozen campaign in
`results/pika-clock-confirmation-plan.json` schedules two independent 200-game
sets per variant, 1,200 games total, without early stopping or retuning between
arms. Planned games are not completed games.

The candidate is the unpromoted expanded small network from the first 8,192-game
iteration with the original attributed jqe search. It is fixed independently of
these sets. It failed its direct-teacher acceptance match; these opponent-specific
tests cannot retroactively establish a general learned improvement.

Both engines receive actual game clocks of 5+0.05 seconds and complete
observer-specific histories, under the pinned independent jqe native referee.
Configuration is one thread and 64 MB hash, with two concurrent games. Each
engine retains its own time manager. The original classical opponents receive
no NNUE file: the runner's required `--net` placeholder is not forwarded when
`--baseline-classical` is set. The candidate does use its recorded NNUE network.

The source checkouts are unmodified and pinned, and binary, candidate-network,
runner and referee-entry hashes are frozen. The launcher checks all artifacts
and source revisions before each arm. Original Pika internal repetition or
chance-search approximations are retained; results concern the stated external
referee profile, not every deployment's rule variant.

A separate two-game actual-clock smoke for the original variant completed with
no errors; all 107 moves, identities, terminal outcomes, and rounded actual clock
charges replayed successfully. This checks the interface, not playing strength,
and its layouts are excluded from confirmation. A clean 100-pair set must meet
both paired bootstrap and Hoeffding lower bounds above 50%; any technical error
or infrastructure void prevents acceptance. All fixed sets must finish, including
negative outcomes. No opponent is marked defeated based on the launch or smoke.
