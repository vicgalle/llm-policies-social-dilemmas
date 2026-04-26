"""Frozen cooperator registry — π_B^coop per env (arimd_plan.md §3.3).

Each entry exposes:
- ``fn`` — the policy callable, signature ``(env, agent_id) -> int``.
- ``code`` — the source code string (shown to Red and Blue prompts).
- ``name`` — human label used in run logs.

For ARIMD, cooperators are intentionally hand-crafted and stable across
runs. They are *not* the synthesizer's output. If a paper run wants to
swap in a different cooperator (e.g., to re-test H2), add it here.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Tuple


@dataclass(frozen=True)
class Cooperator:
    name: str
    fn: Callable
    code: str
    description: str


def _read_module_source(mod) -> str:
    try:
        return inspect.getsource(mod)
    except Exception:
        return ""


def _build_registry() -> Dict[Tuple[str, str], Cooperator]:
    registry: Dict[Tuple[str, str], Cooperator] = {}

    # nested_commons — efficiency cooperator
    import nested_commons_policy as _nc_eff
    registry[("nested_commons", "efficiency")] = Cooperator(
        name="nc_efficiency",
        fn=_nc_eff.policy,
        code=_read_module_source(_nc_eff),
        description=(
            "Efficiency-tuned nested_commons cooperator. Static role split: "
            "agent_id % 4 ∈ {0=river_cleaner, 1/2=harvesters, 3=plaza_specialist}. "
            "Never raids or gifts. U≈15, maximin≈500 on the post-retune env."
        ),
    )

    # nested_commons — maximin cooperator
    import nested_commons_maximin_policy as _nc_mm
    registry[("nested_commons", "maximin")] = Cooperator(
        name="nc_maximin",
        fn=_nc_mm.policy,
        code=_read_module_source(_nc_mm),
        description=(
            "Maximin-tuned nested_commons cooperator. Same instantaneous role mix "
            "as the efficiency cooperator but rotates every agent through every "
            "role on a 4×4 Latin square every 250 steps. U≈14.7, maximin≈817."
        ),
    )

    # production_economy — default cooperator
    import production_economy_policy as _pe
    registry[("production_economy", "default")] = Cooperator(
        name="pe_default",
        fn=_pe.policy,
        code=_read_module_source(_pe),
        description=(
            "Three-phase production_economy cooperator: tool acquisition → "
            "shelter rush before winter → post-winter steady-state tool cycling."
        ),
    )

    # cleanup — minimal hand-crafted cooperator
    import cleanup_policy as _cu
    registry[("cleanup", "default")] = Cooperator(
        name="cu_default",
        fn=_cu.policy,
        code=_read_module_source(_cu),
        description=(
            "Adaptive role-split cleanup cooperator. ~20% of agents act as "
            "cleaners; surge to ~40% if waste density >= 0.32; rest as "
            "harvesters when waste <= 0.05. Never fires the BEAM."
        ),
    )

    return registry


_REGISTRY = _build_registry()


def get_cooperator(game: str, name: str) -> Cooperator:
    key = (game, name)
    if key not in _REGISTRY:
        avail = ", ".join(f"{g}:{n}" for (g, n) in _REGISTRY)
        raise ValueError(
            f"No cooperator for game={game!r}, name={name!r}. Available: {avail}"
        )
    return _REGISTRY[key]


def list_cooperators(game: str | None = None):
    if game is None:
        return sorted(_REGISTRY.keys())
    return sorted(n for (g, n) in _REGISTRY if g == game)


if __name__ == "__main__":
    print("Cooperator registry:")
    for (g, n), coop in _REGISTRY.items():
        print(f"  {g:24s} {n:12s} {coop.name:18s} ({len(coop.code)} chars of source)")
