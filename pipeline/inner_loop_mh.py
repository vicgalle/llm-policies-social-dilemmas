"""
Metropolis–Hastings-accepted inner loop (secondary PGA contribution).

Replaces the default "regenerate-from-scratch" inner-loop iteration with a
STOKE-style local mutation + MH acceptance step. Purpose: **variance
containment** — one bad regeneration can't discard hard-won structure,
because the MH rule keeps the previous accepted policy unless the new
proposal improves Φ.

One MH iteration (pseudo-code, see plan §MH-Accepted Inner Loop):

    1. Start from the current best policy π_{k-1} (with its Φ, profile).
    2. LLM proposes a **local mutation** m(π_{k-1}), guided by the profile
       P — e.g. "branch X never fires under the current guard; relax the
       guard or add a triggering condition".
    3. Evaluate Φ(π_k) over a small seed set (same as the default loop).
    4. Accept with probability min(1, exp(β_k (Φ_k − Φ_{k-1}))) where β_k
       is annealed linearly from ``MH_BETA_0`` to ``MH_BETA_FINAL``.
    5. If rejected, keep π_{k-1} for the next iteration; otherwise replace.

The MH step composes cleanly with PGA: the profile both *targets the
mutation* (step 2) and *diagnoses the rejection* (step 5, for the next
iteration's prompt).

This module is wired into ``run_inner_loop.py`` only when
``config.MH_INNER_LOOP`` is True (or the corresponding CLI flag is passed).
When False, the default regenerate-from-scratch loop is used.
"""
from __future__ import annotations

import math
import random
import textwrap
import time
from dataclasses import dataclass
from typing import Callable, Optional

from . import config as _cfg


@dataclass
class MHState:
    """The MH chain's current accepted candidate."""
    code: str
    fn: Callable
    name: str
    reasoning: str
    phi: float              # the accepted Φ (primary-metric value)
    metrics: dict
    profile: Optional[dict]
    profile_markdown: Optional[str]


def beta_schedule(k: int, k_total: int,
                  beta_0: Optional[float] = None,
                  beta_final: Optional[float] = None) -> float:
    """Linear annealing of β over the inner loop.

    ``k`` is 1-indexed (the first proposed mutation is k=1). At k=k_total
    we reach ``MH_BETA_FINAL``.
    """
    b0 = beta_0 if beta_0 is not None else _cfg.MH_BETA_0
    bf = beta_final if beta_final is not None else _cfg.MH_BETA_FINAL
    if k_total <= 1:
        return bf
    frac = max(0.0, min(1.0, (k - 1) / (k_total - 1)))
    return b0 + frac * (bf - b0)


def accept(phi_new: float, phi_old: float, beta: float,
           rng: Optional[random.Random] = None) -> bool:
    """Metropolis–Hastings acceptance for a maximisation objective."""
    if phi_new >= phi_old:
        return True
    log_p = beta * (phi_new - phi_old)
    if log_p < -50:
        return False
    p = math.exp(log_p)
    r = rng.random() if rng is not None else random.random()
    return r < p


def build_mutation_prompt(state: MHState, iteration: int, k_total: int,
                          base_prompt_tail: str) -> str:
    """Turn a base iteration prompt into a local-mutation proposal prompt.

    The policy LLM is asked for a *small edit* to the current accepted
    policy rather than a full regeneration. The profile-driven hints that
    the default builder already injects remain in ``base_prompt_tail`` so
    we don't duplicate them.
    """
    return textwrap.dedent(f"""\
    ## Iteration {iteration}/{k_total}: Propose a LOCAL MUTATION (MH mode)

    The inner loop is running in **Metropolis–Hastings mode**. Instead of
    rewriting the policy from scratch, propose the *smallest* targeted
    edit to the accepted policy below that plausibly improves the primary
    metric. Keep all existing structure that is working; change only the
    branch(es) the profile flagged.

    The proposal will be evaluated and then **accepted only if it
    improves the primary metric** (or stochastically at inverse-
    temperature β={beta_schedule(iteration, k_total):.2f}). A regression
    will be rejected — so do not take risks on unrelated refactors.

    ### Accepted policy (the current chain state, Φ = {state.phi:.3f})
    ```python
    {state.code}
    ```

    {base_prompt_tail}
    """)
