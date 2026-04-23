"""
Pipeline configuration — iteration parameters.

The researcher can modify these to change the inner loop behavior:
- Number of iterations K
- Number of evaluation seeds
- Retry budget
- Thinking budget for the policy LLM
- MH inner-loop parameters (variance containment — see pga_improvement_plan.md)
- Any other iteration-level hyperparameters
"""

# Number of inner-loop iterations (policy refinement steps).
# The policy LLM generates K+1 policies (iteration 0 = from scratch, then K refinements).
N_ITERATIONS = 3

# Number of random seeds for evaluation. More seeds = less variance but more cost.
N_EVAL_SEEDS = 5

# Maximum retries when the policy LLM generates invalid code.
MAX_RETRIES = 3

# Extended thinking token budget (for Claude models with thinking support).
THINKING_BUDGET = 16000

# Evaluation timeout per matchup (seconds).
EVAL_TIMEOUT = 600

# --- Profile-Guided Autoresearch (PGA) ------------------------------------

# Number of time-buckets to split the horizon into for the action histogram
# profile channel. 3 = thirds (matches the paper default).
N_PROFILE_BUCKETS = 3

# --- MH-accepted inner loop (variance containment) ------------------------
# When enabled, each inner-loop iteration proposes a LOCAL MUTATION of the
# current best policy rather than regenerating from scratch. Accepts with
# probability min(1, exp(beta * (Φ_new - Φ_old))) with β annealed across
# the loop. Off by default; flip to True (and set --inner-loop mh in the
# CLI if added) to run Experiment 1's "PGA + MH" arm.
MH_INNER_LOOP = False
MH_BETA_0 = 1.0           # initial inverse-temperature (linear schedule)
MH_BETA_FINAL = 5.0       # β at the last inner iteration
MH_METRIC = "efficiency"  # which channel drives acceptance (efficiency or maximin)
