"""
Meta-Harness abstraction.

A *harness* is the artifact produced by the proposer. It is mode-typed:

    M1: feedback-shaper  — a copy of pipeline/{prompts,feedback,helpers,
                            config}.py whose synthesis run produces a final
                            policy callable. The synthesis is performed once
                            at harness creation; the resulting policy is
                            cached at harness/synthesized_policy.py and
                            replayed for evaluation.
    M2: scaffold-supplier — a Python scaffold (harness/scaffold.py exposing
                            ``make_policy(base_model: str | None) -> callable``).
                            The returned callable may invoke ``base_model`` for
                            high-level decisions; the scaffold handles
                            mechanics in code.
    M3: policy-as-code   — harness/policy.py exposing
                            ``policy(env, agent_id) -> int``. No LLM at
                            inference time.

All three modes ultimately yield a Python callable ``fn(env, agent_id) -> int``
that the meta-eval loop runs in self-play. This module provides:

  * ``Harness`` (abstract): contract for loading and exposing the callable.
  * ``M1FeedbackShaper`` / ``M2Scaffold`` / ``M3PolicyAsCode``: concrete
    classes; ``load_harness`` dispatches based on the manifest's ``mode``.
  * ``run_meta_episode``: the unified episode runner used by the evaluator.
    Records per-step state to a :class:`pipeline.trace.TraceRecorder` when
    one is supplied.

Frozen infrastructure: NOT modifiable by the proposer.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from llm_self_play import (
    GameConfig,
    CLEANUP_CONFIG,
    GATHERING_CONFIG,
    COOP_MINING_CONFIG,
    PRODUCTION_ECONOMY_CONFIG,
    load_policy,
    smoke_test_policy,
)
from cleanup_env import make_cleanup
from gathering_env import make_gathering, make_gathering_large
from coop_mining_env import make_coop_mining, make_coop_mining_large
from production_economy_env import (
    Action as PEAction,
    make_production_economy,
)


# ---------------------------------------------------------------------------
# Env factory + action-name resolution (env-typed)
# ---------------------------------------------------------------------------


_GAME_CONFIGS = {
    "cleanup": CLEANUP_CONFIG,
    "gathering": GATHERING_CONFIG,
    "coop_mining": COOP_MINING_CONFIG,
    "production_economy": PRODUCTION_ECONOMY_CONFIG,
}


def make_env(manifest: Dict[str, Any]):
    """Build the env factory described by a harness manifest.

    Manifest schema::

        {"env": "production_economy", "n_agents": 8, "map": "default", ...}
    """
    game = manifest["env"]
    n_agents = int(manifest.get("n_agents", 0)) or None
    map_kind = manifest.get("map", "default")

    def factory():
        if game == "cleanup":
            return make_cleanup(n_agents=n_agents or 10, small=(map_kind == "small"))
        if game == "gathering":
            if map_kind == "large":
                return make_gathering_large(n_agents=n_agents or 4)
            return make_gathering(n_agents=n_agents or 4, small=(map_kind == "small"))
        if game == "coop_mining":
            if map_kind == "large":
                return make_coop_mining_large(n_agents=n_agents or 6)
            return make_coop_mining(n_agents=n_agents or 4,
                                    small=(map_kind == "small"))
        if game == "production_economy":
            return make_production_economy(n_agents=n_agents or 8)
        raise ValueError(f"Unknown env {game!r}")

    return factory


def action_names_for(game: str) -> List[str]:
    """Return ordered action-name list for trace pretty-printing."""
    if game == "production_economy":
        return [a.name for a in sorted(PEAction, key=lambda x: x.value)]
    if game == "cleanup":
        from cleanup_env import CleanupAction
        return [a.name for a in sorted(CleanupAction, key=lambda x: x.value)]
    if game in ("gathering",):
        from gathering_env import Action as GAction
        return [a.name for a in sorted(GAction, key=lambda x: x.value)]
    if game == "coop_mining":
        from coop_mining_env import Action as MAction
        return [a.name for a in sorted(MAction, key=lambda x: x.value)]
    return []


# ---------------------------------------------------------------------------
# Harness contract
# ---------------------------------------------------------------------------


@dataclass
class HarnessLoadResult:
    """What ``Harness.load`` returns."""
    fn: Callable                       # policy(env, agent_id) -> int
    base_model: Optional[str]          # base model dependency at INFERENCE time
    inference_token_cost: float        # tokens called per env step (estimate; 0 for M3)
    synthesis_token_cost: float        # tokens spent at HARNESS-CREATION time
    notes: str = ""                    # free-form info for the proposer


class Harness:
    """Abstract base. Concrete subclasses implement ``load``."""

    mode: str = "M?"

    def __init__(self, harness_dir: Path, manifest: Dict[str, Any]):
        self.dir = Path(harness_dir)
        self.manifest = manifest

    @classmethod
    def for_dir(cls, harness_dir: Path) -> "Harness":
        return load_harness(harness_dir)

    def load(self) -> HarnessLoadResult:  # pragma: no cover - abstract
        raise NotImplementedError


def load_harness(harness_dir: Path) -> Harness:
    """Read manifest.json and return a typed Harness instance."""
    harness_dir = Path(harness_dir)
    manifest_path = harness_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest.json in {harness_dir}")
    manifest = json.loads(manifest_path.read_text())
    mode = manifest.get("mode", "M3")
    if mode == "M1":
        return M1FeedbackShaper(harness_dir, manifest)
    if mode == "M2":
        return M2Scaffold(harness_dir, manifest)
    if mode == "M3":
        return M3PolicyAsCode(harness_dir, manifest)
    raise ValueError(f"unknown harness mode {mode!r}")


# ---------------------------------------------------------------------------
# M3: policy-as-code
# ---------------------------------------------------------------------------


def _import_module_from(path: Path, module_name: str):
    """Import an arbitrary .py file as a module without polluting sys.path.

    Used by M3 (policy.py) and M2 (scaffold.py) to load proposer-authored
    code without adding the harness dir to sys.path globally.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


class M3PolicyAsCode(Harness):
    """Pure Python policy. No LLM at inference time."""

    mode = "M3"

    def load(self) -> HarnessLoadResult:
        policy_path = self.dir / "harness" / "policy.py"
        if not policy_path.exists():
            raise FileNotFoundError(f"M3 harness missing harness/policy.py "
                                    f"at {self.dir}")
        # Use a unique module name per harness id so multiple harnesses
        # with the same filename can coexist in sys.modules.
        mod_name = f"_meta_h_{self.manifest.get('id', self.dir.name)}_policy"
        mod = _import_module_from(policy_path, mod_name)
        if not hasattr(mod, "policy"):
            raise AttributeError(f"{policy_path} does not define `policy`")
        fn = mod.policy
        return HarnessLoadResult(
            fn=fn,
            base_model=None,
            inference_token_cost=0.0,
            synthesis_token_cost=float(self.manifest.get(
                "synthesis_token_cost", 0.0)),
            notes="M3: no inference-time LLM.",
        )


# ---------------------------------------------------------------------------
# M2: scaffold-supplier
# ---------------------------------------------------------------------------


class M2Scaffold(Harness):
    """Python scaffold + small LLM-decided strategic loop.

    Scaffold contract: harness/scaffold.py defines

        def make_policy(base_model: str | None) -> Callable[[env, int], int]:
            ...

    The returned callable may invoke ``base_model`` for high-level decisions
    while the scaffold handles low-level mechanics (navigation, validation).
    """

    mode = "M2"

    def load(self) -> HarnessLoadResult:
        scaffold_path = self.dir / "harness" / "scaffold.py"
        if not scaffold_path.exists():
            raise FileNotFoundError(f"M2 harness missing harness/scaffold.py "
                                    f"at {self.dir}")
        mod_name = f"_meta_h_{self.manifest.get('id', self.dir.name)}_scaffold"
        mod = _import_module_from(scaffold_path, mod_name)
        if not hasattr(mod, "make_policy"):
            raise AttributeError(f"{scaffold_path} does not define `make_policy`")
        base_model = self.manifest.get("base_model") or None
        fn = mod.make_policy(base_model)
        if not callable(fn):
            raise TypeError("M2 scaffold.make_policy did not return a callable")
        return HarnessLoadResult(
            fn=fn,
            base_model=base_model,
            inference_token_cost=float(self.manifest.get(
                "inference_token_cost", 0.0)),
            synthesis_token_cost=float(self.manifest.get(
                "synthesis_token_cost", 0.0)),
            notes=f"M2 scaffold; base_model={base_model}",
        )


# ---------------------------------------------------------------------------
# M1: feedback-shaper
# ---------------------------------------------------------------------------


class M1FeedbackShaper(Harness):
    """Existing autoresearch pipeline as a harness.

    Contract: harness/synthesized_policy.py contains the FINAL policy code
    produced by running the M1 synthesis loop (pipeline/* in this harness)
    once. The synthesis is performed by ``synthesize_m1_harness`` (called
    by the proposer's ``spawn_branch`` tool when mode=M1) and the result
    cached so re-evaluation is deterministic.

    The harness directory layout::

        harness/
          pipeline/{prompts,feedback,helpers,config}.py
          synthesized_policy.py     # cached final policy from one run
          synthesis_log.json        # K-iteration trajectory used to derive cost
    """

    mode = "M1"

    def load(self) -> HarnessLoadResult:
        cached = self.dir / "harness" / "synthesized_policy.py"
        if not cached.exists():
            raise FileNotFoundError(
                f"M1 harness {self.dir} missing cached synthesized_policy.py. "
                f"Re-run synthesize_m1_harness or spawn_branch with mode=M1."
            )
        code = cached.read_text()
        game = self.manifest.get("env", "production_economy")
        cfg = _GAME_CONFIGS[game]
        # Use the same load_policy helper the inner loop uses, so the
        # namespace (helpers, env enums) matches.
        fn = load_policy(code, extra_namespace=dict(cfg.extra_namespace),
                         tag=f"m1_{self.manifest.get('id', self.dir.name)}")
        base_model = self.manifest.get("base_model")
        synth_cost = float(self.manifest.get("synthesis_token_cost", 0.0))
        return HarnessLoadResult(
            fn=fn,
            base_model=base_model,
            inference_token_cost=0.0,   # cached policy has no inference cost
            synthesis_token_cost=synth_cost,
            notes=("M1 cached policy synthesized by "
                   f"{base_model} (synthesis tokens={synth_cost})"),
        )


# ---------------------------------------------------------------------------
# Unified episode runner with trace hook
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    seed: int
    total_rewards: Dict[int, float]
    metrics: Dict[str, float]
    horizon: int
    n_agents: int
    wall_time_s: float
    error: Optional[str] = None


def run_meta_episode(
    env_factory: Callable,
    fn: Callable,
    seed: int,
    *,
    recorder=None,                       # pipeline.trace.TraceRecorder | None
    max_action: Optional[int] = None,
    on_error: str = "zero",              # "zero" | "raise"
) -> EpisodeResult:
    """Run one self-play episode, all agents using ``fn``.

    Mirrors gathering_policy.run_episode's core loop but:

    * is environment-agnostic (works on any of the four envs by duck-typing
      ``compute_metrics`` and step semantics);
    * optionally records per-step state to a TraceRecorder;
    * returns a plain dataclass instead of a dict (typed callers).

    The env-specific ``compute_metrics`` is invoked via the env class —
    each env exposes it as a static method with the same signature.
    """
    import numpy as np

    env = env_factory()
    env.reset(seed=seed)
    horizon = int(env.max_steps)
    n = int(env.n_agents)

    ep_rewards = {i: [] for i in range(n)}
    ep_timeouts = {i: [] for i in range(n)}

    t0 = time.time()
    error_msg: Optional[str] = None

    for step in range(horizon):
        actions: Dict[int, int] = {}
        try:
            for i in range(n):
                a = fn(env, i)
                a_int = int(a)
                if max_action is not None and not (0 <= a_int <= max_action):
                    raise ValueError(
                        f"policy returned out-of-range action {a_int} "
                        f"for agent {i} at step {step} (max {max_action})"
                    )
                actions[i] = a_int
            obs, rewards, terminated, truncated, info = env.step(actions)
        except Exception as e:
            error_msg = f"runtime error at step {step}: {e}"
            if on_error == "raise":
                raise
            # fill remaining steps with zeros so the metric computation runs.
            for i in range(n):
                ep_rewards[i].append(0.0)
                ep_timeouts[i].append(False)
            for fill_step in range(step + 1, horizon):
                for i in range(n):
                    ep_rewards[i].append(0.0)
                    ep_timeouts[i].append(False)
            break

        for i in range(n):
            ep_rewards[i].append(float(rewards[i]))
            ep_timeouts[i].append(bool(info[i].get("timeout", 0) > 0))

        if recorder is not None:
            recorder.record_step(step + 1, env, actions, rewards, info)

    metrics = type(env).compute_metrics(ep_rewards, ep_timeouts)
    totals = {i: float(sum(ep_rewards[i])) for i in range(n)}

    return EpisodeResult(
        seed=int(seed),
        total_rewards=totals,
        metrics=metrics,
        horizon=horizon,
        n_agents=n,
        wall_time_s=round(time.time() - t0, 3),
        error=error_msg,
    )


# ---------------------------------------------------------------------------
# Smoke validation (used by tools.py spawn / eval to fail fast)
# ---------------------------------------------------------------------------


def validate_harness(harness_dir: Path) -> tuple[bool, str]:
    """Load harness and run a short smoke episode. Returns (ok, msg)."""
    try:
        h = load_harness(harness_dir)
        loaded = h.load()
        env_factory = make_env(h.manifest)
        game = h.manifest.get("env", "production_economy")
        max_action = _GAME_CONFIGS[game].max_action
        passed, err = smoke_test_policy(loaded.fn, env_factory=env_factory,
                                        max_action=max_action, n_steps=20)
        if not passed:
            return False, err
        return True, "smoke ok"
    except Exception as e:  # surfacing the raw error is fine — proposer reads it
        import traceback
        return False, f"{e}\n{traceback.format_exc()}"
