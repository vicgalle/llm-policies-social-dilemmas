"""Iteration parameters for the M1 synthesizer.

Tuned for synthesis-time cost rather than for committing-to-Pareto.
Synthesis only needs a *signal*, not a precise per-iteration metric —
the framework re-evals the cached policy with the adaptive evaluator
(2/5/20 seeds) anyway. So small N_EVAL_SEEDS here is fine.
"""

# Total inner-loop iterations (the synthesizer generates K+1 policies:
# iteration 0 from scratch, then K refinements informed by the
# accumulated history).
N_ITERATIONS = 2

# Self-play eval seeds per iteration during synthesis. The proposer's
# spawn pays this cost; keep it small.
N_EVAL_SEEDS = 3

# Retry budget when the synthesizer emits invalid code.
MAX_RETRIES = 3
