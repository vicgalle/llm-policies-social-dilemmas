"""
Pipeline configuration — iteration parameters.

The researcher can modify these to change the inner loop behavior:
- Number of iterations K
- Number of evaluation seeds
- Retry budget
- Thinking budget for the policy LLM
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
