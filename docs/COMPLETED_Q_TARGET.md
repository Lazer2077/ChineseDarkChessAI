# Completed-Q policy-target feasibility diagnostic

Search beats the raw warm policy, yet distilling exploratory visit distributions
can regress. The constant policy anchor recovers part of that regression but
does not beat the original warm model. This motivates examining the supervision
target itself, independently of another training campaign.

[Danihelka et al., ICLR2022](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf)
derive a target from completed action values rather than exploration counts.
Equation11tilts the old logits by scaled completed Q; AppendixD's Equation33
fills unvisited actions with a mixture of root network value and the
prior-weighted mean Q over visited actions. The visited mean uses policy
weights, not visit-count weights. Their guarantees cannot simply be transferred
to our approximate neural/particle action values. We implement only this target
transformation, not the full Gumbel root/interior algorithm.

For legal-action prior p, completed visit counts n, root-player Q estimates q
and root network value v:

    q_seen = sum_seen(p_a*q_a) / sum_seen(p_a)
    v_mix = (v + sum(n)*q_seen) / (1 + sum(n))
    q_complete(a) = q_a if n_a>0 else v_mix
    target = softmax(log(p) + scale*q_complete).

No visits returns the prior. Zero prior support stays zero; unknown values are
permitted only on unvisited actions. The sign is explicitly root-player Q,
opposite the existing edge's next-player Q. A positive gain against this same
completed vector is an algebraic property of an exponential tilt, **not evidence
of improved true values, teacher scores or winning probability**.

Three tests cover the mixed-value formula with deliberately unequal visit
counts, unchanged priors at zero scale/no visits, support/finite-value checks
and the self-objective relation over100finite vector examples.

## Three reused TRAIN roots, no fitting

Unchanged warm PUCT performs256completed simulations/LeafBatch8without
Dirichlet noise, preserving legal priors before any exploration perturbation.
Scale is0.5*(50+max_visits), consistent with converting bounded rootQ[-1,1]
to the[0,1]convention used by the existing halving prototype. Each target is
computed from exactly the same completed search, with no altered acting.

An independent original native teacher gives depth4forced-move cp scores from
the actor's observation FEN, with cold Hash16MB. **That FEN resets history**:
these are shallow heuristic proxies, not true Q, equilibrium values or actual
full-history game regret. The roots have repeatedly been used for TRAIN
diagnostics and are not independent strength tests.

| Root | Target entropy: visits / completed-Q | Teacher proxy expected cp: prior / visits / completed-Q |
| --- | ---: | ---: |
| Opening | 1.3846 / 0.3709 | 78.183 / 71.426 / 78.755 |
| Uncertain last own dark | 2.0629 / 2.4610 | 2370.782 / 2354.957 / 2370.157 |
| Fully public | 2.2808 / approximately0 | -242.798 / -302.613 / -405.000 |

In the fully public root, the target becomes almost deterministic and raises
its own completed-Q objective by0.6061, yet worsens the separate shallow proxy.
This illustrates the risk of magnifying approximate action values even after
physical hidden-type uncertainty disappears. It does not disprove completed-Q
learning, establish a better scale or demonstrate a failure against actual
game outcomes. No scale sweep or trained completed-Q campaign is launched.

Future use would need truthful recording of root value, **unperturbed** legal
priors and completed action-value sums/counts alongside actual visits. Existing
visit-only episodes cannot be relabeled by inventing missing Q values. A new
controlled data-generation/training recipe and fresh paired matches would be
required. None of the live constant/entropy-anchor source files is changed by
this diagnostic.

Evidence: `results/completed-q-target-diagnostic-plan.json` and
`results/20261027-completed-q-target-diagnostic.json`.
