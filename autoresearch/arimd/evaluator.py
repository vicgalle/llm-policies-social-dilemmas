"""Mixed-population evaluation harness for ARIMD (arimd_plan.md §2.1).

For a given env design (factory) and a Red policy, sample several
``λ``-fraction Red-index assignments × seeds, run each episode, and
aggregate per-population statistics:

- ``v_blue``     — mean per-agent return over Blue indices.
- ``v_red``      — mean per-agent return over Red indices (Red's objective).
- ``v_pop``      — mean per-agent return over the full population.
- ``metrics``    — env-native social metrics averaged over episodes.
- ``raid_attempts``, ``raid_successes`` — flagged for degeneracy detection.

The Stackelberg outer loop's payoff functional ``V_λ(e, π_R)`` is
``v_blue`` by default (Blue-welfare convention; toggle via ``welfare``).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from gathering_policy import run_episode


@dataclass
class MixedEvalResult:
    v_blue: float           # mean per-agent return over Blue indices
    v_red: float            # mean per-agent return over Red indices
    v_pop: float            # mean per-agent return over all agents
    n_episodes: int
    n_blue_per_ep: int
    n_red_per_ep: int
    metrics: Dict[str, float] = field(default_factory=dict)
    raid_attempts: float = 0.0       # mean per-episode raid attempts (or 0 if env has no raids)
    raid_successes: float = 0.0
    eval_time: float = 0.0
    per_episode: List[Dict[str, Any]] = field(default_factory=list)

    def primary(self, welfare: str = "blue") -> float:
        """V_λ used by the Stackelberg loop."""
        if welfare == "blue":
            return self.v_blue
        if welfare == "population":
            return self.v_pop
        if welfare == "red":
            return self.v_red
        raise ValueError(f"unknown welfare={welfare!r}")


def _assign_red_indices(
    n_agents: int, n_red: int, rng: np.random.Generator
) -> List[int]:
    if n_red <= 0:
        return []
    if n_red >= n_agents:
        return list(range(n_agents))
    return sorted(rng.choice(n_agents, size=n_red, replace=False).tolist())


def _episode_raid_stats(env) -> Tuple[int, int]:
    """Return (raid_attempts_total, raid_successes_total) if the env tracks them."""
    a = int(getattr(env, "_raid_attempts_total", 0))
    s = int(getattr(env, "_raid_successes_total", 0))
    return a, s


def evaluate_mixed(
    env_factory: Callable,
    blue_fn: Callable,
    red_fn: Optional[Callable],
    lambda_: float,
    seeds: List[int],
    *,
    n_assignments: int = 1,
    welfare: str = "blue",                   # unused here — preserved for API symmetry
    rng_seed: int = 12345,
    fail_on_red_error: bool = False,
) -> MixedEvalResult:
    """Run ``λ``-mixed-population episodes and return aggregated stats.

    Parameters
    ----------
    env_factory : callable () -> env
        Factory that returns a fresh env. Each episode resets internally.
    blue_fn : callable (env, agent_id) -> int
        The frozen cooperator policy.
    red_fn : callable or None
        The exploit policy. If ``None`` or ``λ == 0``, runs pure Blue
        self-play (V_0 baseline).
    lambda_ : float in [0, 1]
        Fraction of agents that play Red.
    seeds : list of ints
        Eval seeds. The same seeds are used across all assignments per
        seed for variance control.
    n_assignments : int
        Number of Red-index reshuffles per seed. Default 1 — one
        assignment per seed.
    rng_seed : int
        Reproducibility seed for assignment sampling.
    fail_on_red_error : bool
        If True, propagate exceptions from Red. If False (default),
        treat a failing Red as a no-op (STAND/NOOP) and record it.
    """
    t0 = time.time()
    rng = np.random.default_rng(rng_seed)

    # Probe one env to learn n_agents / max_action.
    probe = env_factory()
    n_agents = int(probe.n_agents)
    if hasattr(probe, "action_space_n"):
        max_action_value = int(probe.action_space_n) - 1
    else:
        max_action_value = 1_000_000
    del probe

    n_red = int(round(float(lambda_) * n_agents))
    if red_fn is None:
        n_red = 0

    per_blue_returns: List[float] = []
    per_red_returns: List[float] = []
    per_pop_returns: List[float] = []
    metric_accum: Dict[str, List[float]] = {}
    raid_attempts: List[int] = []
    raid_successes: List[int] = []
    per_episode: List[Dict[str, Any]] = []

    def _safe_red(env, aid: int) -> int:
        try:
            a = int(red_fn(env, aid))
        except Exception:
            if fail_on_red_error:
                raise
            return 0  # NOOP-equivalent for all envs (STAND/NOOP both encode to 7 or 0)
        if not (0 <= a <= max_action_value):
            return 0
        return a

    for seed in seeds:
        for asg_i in range(n_assignments):
            red_ids = set(_assign_red_indices(n_agents, n_red, rng))
            blue_ids = set(range(n_agents)) - red_ids

            agent_fns: Dict[int, Callable] = {}
            for i in range(n_agents):
                if i in red_ids and red_fn is not None:
                    agent_fns[i] = _safe_red
                else:
                    agent_fns[i] = blue_fn

            env = env_factory()
            try:
                result = run_episode(
                    env,
                    agent_fns=agent_fns,
                    seed=int(seed),
                    verbose=False,
                )
            except Exception as e:
                # Preserve the run; record a zeroed entry.
                per_episode.append({
                    "seed": int(seed),
                    "assignment": asg_i,
                    "red_ids": sorted(red_ids),
                    "error": repr(e),
                })
                continue

            totals = result["total_rewards"]
            metrics = result["metrics"]

            blue_rs = [totals[i] for i in blue_ids]
            red_rs = [totals[i] for i in red_ids]
            pop_rs = list(totals.values())

            per_blue_returns.append(float(np.mean(blue_rs)) if blue_rs else 0.0)
            per_red_returns.append(float(np.mean(red_rs)) if red_rs else 0.0)
            per_pop_returns.append(float(np.mean(pop_rs)) if pop_rs else 0.0)

            for k, v in metrics.items():
                metric_accum.setdefault(k, []).append(float(v))

            ra, rs = _episode_raid_stats(env)
            raid_attempts.append(ra)
            raid_successes.append(rs)

            per_episode.append({
                "seed": int(seed),
                "assignment": asg_i,
                "red_ids": sorted(red_ids),
                "v_blue": per_blue_returns[-1],
                "v_red": per_red_returns[-1],
                "v_pop": per_pop_returns[-1],
                "metrics": metrics,
                "raid_attempts": ra,
                "raid_successes": rs,
            })

    n_eps = len(per_blue_returns)
    if n_eps == 0:
        return MixedEvalResult(
            v_blue=0.0, v_red=0.0, v_pop=0.0,
            n_episodes=0,
            n_blue_per_ep=n_agents - n_red,
            n_red_per_ep=n_red,
            metrics={},
            eval_time=time.time() - t0,
            per_episode=per_episode,
        )

    return MixedEvalResult(
        v_blue=float(np.mean(per_blue_returns)),
        v_red=float(np.mean(per_red_returns)),
        v_pop=float(np.mean(per_pop_returns)),
        n_episodes=n_eps,
        n_blue_per_ep=n_agents - n_red,
        n_red_per_ep=n_red,
        metrics={k: float(np.mean(v)) for k, v in metric_accum.items()},
        raid_attempts=float(np.mean(raid_attempts)) if raid_attempts else 0.0,
        raid_successes=float(np.mean(raid_successes)) if raid_successes else 0.0,
        eval_time=time.time() - t0,
        per_episode=per_episode,
    )


# ---------------------------------------------------------------------------
# Factory builder — translates (game, env_kwargs) into an env_factory.
# ---------------------------------------------------------------------------


def make_env_factory(game: str, env_kwargs: Dict[str, Any]) -> Callable:
    """Return a zero-arg factory producing a fresh env with the given kwargs.

    Each call returns a brand-new env instance (no shared state). The
    caller is responsible for ``env.reset(seed=...)`` (run_episode does
    this internally).
    """
    if game == "nested_commons":
        from nested_commons_env import make_nested_commons

        def factory():
            return make_nested_commons(**env_kwargs)
        return factory

    if game == "cleanup":
        from cleanup_env import make_cleanup

        def factory():
            return make_cleanup(**env_kwargs)
        return factory

    if game == "production_economy":
        from production_economy_env import make_production_economy

        def factory():
            return make_production_economy(**env_kwargs)
        return factory

    raise ValueError(f"Unknown game={game!r}")


def merge_env_kwargs(
    game: str,
    grammar_kwargs: Dict[str, Any],
    n_agents: Optional[int] = None,
) -> Dict[str, Any]:
    """Combine grammar-derived kwargs with population-size override."""
    out = dict(grammar_kwargs)
    if n_agents is not None:
        out["n_agents"] = int(n_agents)
    return out
