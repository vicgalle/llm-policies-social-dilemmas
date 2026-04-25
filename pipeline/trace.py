"""
Meta-Harness execution trace recorder.

Frozen infrastructure (NOT modifiable by the proposer). The proposer reads
the JSONL traces and the aggregated summary as part of its diagnostic
context per harness — the "10M-token diagnostic footprint" of Lee et al.
2026 (Meta-Harness).

Per-step record (one JSON object per line in <seed>.jsonl):

    {
      "t": <step>,
      "agents": [{"id", "pos", "inv"?, "tool"?, "tool_age"?, "action",
                  "action_name", "reward", "timeout"?, "beam_hits"?, "orient"?}, ...],
      "globals": {<env-specific fields when present>},
      "events": []          # populated by harness via record_event
    }

Aggregated summary (summary.json), written on finalize():

    {
      "n_agents", "horizon",
      "agent_totals", "rewards_per_step",
      "action_counts": {agent_id: {action_name: count}},
      "tool_history": {agent_id: [[start, end|"end"], ...]},   # if env has has_tool
      "deadlock_events": [{"agent", "t", "duration", "cell", "blocked_by"}],
      "no_op_action_rate": {agent_id: float},
      "extras": {... env-specific aggregates ...}
    }

The recorder is environment-agnostic by duck-typing the env. Production
Economy receives a few extra aggregates (forge contention, pool
saturation events) when the corresponding attributes exist; otherwise
those aggregates are silently skipped.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Per-step extraction helpers
# ---------------------------------------------------------------------------


def _agent_state(env, agent_id: int) -> Dict[str, Any]:
    """Return a per-agent state snapshot, picking optional fields by hasattr."""
    state: Dict[str, Any] = {
        "id": int(agent_id),
        "pos": [int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1])],
    }
    if hasattr(env, "agent_orient"):
        state["orient"] = int(env.agent_orient[agent_id])
    if hasattr(env, "inventory"):
        state["inv"] = [int(x) for x in env.inventory[agent_id]]
    if hasattr(env, "has_tool"):
        state["tool"] = bool(env.has_tool[agent_id])
    if hasattr(env, "tool_age"):
        state["tool_age"] = int(env.tool_age[agent_id])
    if hasattr(env, "agent_timeout"):
        state["timeout"] = int(env.agent_timeout[agent_id])
    if hasattr(env, "agent_beam_hits"):
        state["beam_hits"] = int(env.agent_beam_hits[agent_id])
    return state


def _env_globals(env) -> Dict[str, Any]:
    """Capture env-specific global state via duck-typing."""
    g: Dict[str, Any] = {}
    if hasattr(env, "shelter_count"):
        g["shelter_count"] = int(env.shelter_count)
    if hasattr(env, "_step_count"):
        g["step"] = int(env._step_count)
    # Production Economy: forge pool snapshot ([planks, bricks] per forge).
    if hasattr(env, "forge_cells_list") and hasattr(env, "dropped_items"):
        try:
            from production_economy_env import PLANK as _P, BRICK as _B
            forges = []
            for r, c in env.forge_cells_list:
                forges.append({
                    "cell": [int(r), int(c)],
                    "planks": int(env.dropped_items[r, c, _P]),
                    "bricks": int(env.dropped_items[r, c, _B]),
                    "total": int(env.dropped_items[r, c].sum()),
                })
            g["forge_pools"] = forges
        except Exception:
            pass
    # Cleanup / Gathering: alive apple count.
    if hasattr(env, "apple_alive"):
        try:
            g["n_apples_alive"] = int(env.apple_alive.sum())
        except Exception:
            pass
    if hasattr(env, "waste_count"):
        g["waste_count"] = int(env.waste_count)
    elif hasattr(env, "n_waste"):
        try:
            g["waste_count"] = int(env.n_waste)
        except Exception:
            pass
    return g


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class TraceRecorder:
    """Stream per-step env state to JSONL and aggregate a summary.

    Usage from a harness episode runner:

        rec = TraceRecorder(path, env_meta={...}, action_names=[...])
        rec.record_step(t, env, actions, rewards)
        ...
        rec.finalize()                # writes summary.json next to the jsonl

    The recorder writes per-step lines incrementally so even crashed
    episodes leave partial traces the proposer can inspect.
    """

    DEADLOCK_MIN_STEPS = 10  # consecutive same-position despite a move action

    def __init__(
        self,
        path: Path | str,
        *,
        action_names: Optional[List[str]] = None,
        env_meta: Optional[Dict[str, Any]] = None,
        compact: bool = True,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.action_names = action_names or []
        self.env_meta = dict(env_meta or {})
        self.compact = compact
        self._fh = self.path.open("w")

        # Header line: meta about the rollout (so the trace is
        # self-describing without a sidecar).
        header = {
            "_header": True,
            "env": self.env_meta,
            "action_names": self.action_names,
        }
        self._write(header)

        # Aggregation state -------------------------------------------------
        self._n_steps = 0
        self._n_agents: int | None = None
        self._agent_totals: Dict[int, float] = defaultdict(float)
        self._rewards_per_step: List[float] = []  # collective reward per step
        self._action_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._move_action_names = {"MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W",
                                   "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT"}

        # Tool-history bookkeeping (per agent: list of [start, end_step|"end"]).
        self._tool_history: Dict[int, List[list]] = defaultdict(list)
        self._tool_open: Dict[int, Optional[int]] = defaultdict(lambda: None)

        # Deadlock tracking — same-pos despite move action.
        self._stuck_run: Dict[int, Dict[str, Any]] = {}
        self._deadlock_events: List[Dict[str, Any]] = []

        # PE forge contention counters.
        self._forge_contention_steps = 0

        # Pending events written by the harness via record_event.
        self._pending_events: List[Dict[str, Any]] = []

        # Last seen state to detect tool transitions / deadlocks across calls.
        self._prev_pos: Dict[int, tuple] = {}
        self._prev_has_tool: Dict[int, bool] = {}

    # ------------------------------------------------------------------
    # Internal IO
    # ------------------------------------------------------------------

    def _write(self, obj: Dict[str, Any]) -> None:
        # Compact serialization keeps trace files dense; the proposer
        # treats them as line-oriented logs (one event per line).
        sep = (",", ":") if self.compact else (",", ": ")
        self._fh.write(json.dumps(obj, separators=sep))
        self._fh.write("\n")

    # ------------------------------------------------------------------
    # Public hooks called from the episode runner
    # ------------------------------------------------------------------

    def record_event(self, t: int, kind: str, payload: Dict[str, Any]) -> None:
        """Buffer a structured event for inclusion in the next step record.

        Episode runners can call this between record_step calls to flag
        things like "agent 3 attempted DROP_BRICK on a full forge" that
        the env doesn't surface as a reward signal but matter for the
        proposer's diagnosis.
        """
        evt = {"agent": payload.get("agent"), "kind": kind, **payload}
        evt.setdefault("t", t)
        self._pending_events.append(evt)

    def record_step(
        self,
        t: int,
        env,
        actions: Dict[int, int],
        rewards: Dict[int, float],
        info: Optional[Dict[int, dict]] = None,
    ) -> None:
        """Record one full step of self-play. Call AFTER env.step()."""
        n = env.n_agents
        if self._n_agents is None:
            self._n_agents = n

        agents_block: List[Dict[str, Any]] = []
        forge_set = getattr(env, "forge_cells_set", None)
        forge_occupants = 0

        step_collective_reward = 0.0

        for i in range(n):
            ast = _agent_state(env, i)
            a_int = int(actions.get(i, 0))
            ast["action"] = a_int
            ast["action_name"] = (
                self.action_names[a_int]
                if 0 <= a_int < len(self.action_names) else f"a{a_int}"
            )
            r = float(rewards.get(i, 0.0))
            ast["reward"] = r
            agents_block.append(ast)

            # Aggregations -------------------------------------------------
            self._agent_totals[i] += r
            step_collective_reward += r
            self._action_counts[i][ast["action_name"]] += 1

            # Tool transitions
            if "tool" in ast:
                cur = bool(ast["tool"])
                prev = self._prev_has_tool.get(i, False)
                if cur and not prev:
                    self._tool_open[i] = t
                elif not cur and prev and self._tool_open.get(i) is not None:
                    self._tool_history[i].append([self._tool_open[i], t])
                    self._tool_open[i] = None
                self._prev_has_tool[i] = cur

            # Deadlock detection: if the agent issued a move action and the
            # position did not change, start/extend a stuck run.
            cur_pos = tuple(ast["pos"])
            prev_pos = self._prev_pos.get(i)
            move_attempted = ast["action_name"] in self._move_action_names
            if move_attempted and prev_pos is not None and cur_pos == prev_pos:
                run = self._stuck_run.get(i)
                if run is None:
                    self._stuck_run[i] = {
                        "agent": i,
                        "t_start": t - 1,
                        "cell": list(cur_pos),
                        "duration": 1,
                    }
                else:
                    run["duration"] += 1
            else:
                run = self._stuck_run.pop(i, None)
                if run is not None and run["duration"] >= self.DEADLOCK_MIN_STEPS:
                    # Snapshot blockers (other agents adjacent to the stuck cell).
                    blockers = []
                    sr, sc = run["cell"]
                    for j in range(n):
                        if j == i:
                            continue
                        jr = int(env.agent_pos[j, 0])
                        jc = int(env.agent_pos[j, 1])
                        if abs(jr - sr) + abs(jc - sc) <= 1:
                            blockers.append(j)
                    run["blocked_by"] = blockers
                    run["t_end"] = t
                    self._deadlock_events.append(run)
            self._prev_pos[i] = cur_pos

            # Forge contention
            if forge_set is not None and cur_pos in forge_set:
                forge_occupants += 1

        if forge_set is not None and forge_occupants >= 2:
            self._forge_contention_steps += 1

        rec = {
            "t": int(t),
            "agents": agents_block,
            "globals": _env_globals(env),
        }
        if self._pending_events:
            rec["events"] = self._pending_events
            self._pending_events = []
        self._write(rec)
        self._n_steps += 1
        self._rewards_per_step.append(step_collective_reward)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> Dict[str, Any]:
        """Close the JSONL and write a summary.json sibling. Returns summary."""
        # Close any still-open tool intervals.
        for i, t0 in list(self._tool_open.items()):
            if t0 is not None:
                self._tool_history[i].append([t0, "end"])
                self._tool_open[i] = None

        # Flush any final stuck run that crossed the threshold.
        for i, run in list(self._stuck_run.items()):
            if run["duration"] >= self.DEADLOCK_MIN_STEPS:
                run.setdefault("blocked_by", [])
                run.setdefault("t_end", self._n_steps)
                self._deadlock_events.append(run)

        n_steps = max(self._n_steps, 1)
        no_op_rate: Dict[int, float] = {}
        for i, counts in self._action_counts.items():
            no_op = counts.get("NOOP", 0) + counts.get("STAND", 0)
            no_op_rate[i] = round(no_op / n_steps, 4)

        summary = {
            "env": self.env_meta,
            "n_agents": self._n_agents,
            "horizon": self._n_steps,
            "agent_totals": [round(self._agent_totals[i], 2)
                             for i in range(self._n_agents or 0)],
            "rewards_per_step": [round(x, 3) for x in self._rewards_per_step],
            "action_counts": {int(k): dict(v) for k, v in self._action_counts.items()},
            "tool_history": {int(k): v for k, v in self._tool_history.items()},
            "deadlock_events": self._deadlock_events,
            "no_op_action_rate": no_op_rate,
            "extras": {
                "forge_contention_steps": self._forge_contention_steps,
            },
        }

        self._fh.flush()
        self._fh.close()

        summary_path = self.path.with_suffix("").with_name(
            self.path.stem + ".summary.json"
        )
        # Sibling-style: <stem>.summary.json next to <stem>.jsonl. The
        # per-rollout summary file lives next to its trace; per-harness
        # aggregation across seeds is computed by tools.py at read time.
        summary_path.write_text(json.dumps(summary, indent=2))
        return summary

    # ------------------------------------------------------------------
    # Context manager sugar (so callers can `with TraceRecorder(...) as rec`)
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Finalize even on exception so partial traces still produce a
        # summary the proposer can read.
        try:
            self.finalize()
        except Exception:
            try:
                self._fh.close()
            except Exception:
                pass
        return False  # don't suppress exceptions


# ---------------------------------------------------------------------------
# Aggregation across seeds (for summary.json at the harness level)
# ---------------------------------------------------------------------------


def aggregate_seed_summaries(seed_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine per-seed summaries into a single per-harness summary.

    Aggregates we want for the proposer to grep without reading per-step
    JSONL: averaged action counts, total deadlock counts, per-agent total
    means, episode-level extras.
    """
    if not seed_summaries:
        return {}
    n_agents = seed_summaries[0]["n_agents"]
    out: Dict[str, Any] = {
        "n_seeds": len(seed_summaries),
        "n_agents": n_agents,
        "horizon": seed_summaries[0]["horizon"],
        "agent_totals_mean": [
            round(sum(s["agent_totals"][i] for s in seed_summaries) / len(seed_summaries), 2)
            for i in range(n_agents)
        ],
        "agent_totals_min": [
            round(min(s["agent_totals"][i] for s in seed_summaries), 2)
            for i in range(n_agents)
        ],
        "agent_totals_max": [
            round(max(s["agent_totals"][i] for s in seed_summaries), 2)
            for i in range(n_agents)
        ],
        "deadlock_events_total": sum(
            len(s.get("deadlock_events", [])) for s in seed_summaries
        ),
        "forge_contention_steps_mean": round(
            sum(s.get("extras", {}).get("forge_contention_steps", 0)
                for s in seed_summaries) / len(seed_summaries), 1
        ),
    }
    # Mean per-agent action counts (action_name -> mean across seeds).
    action_mean: Dict[str, float] = defaultdict(float)
    for s in seed_summaries:
        for _aid, counts in s.get("action_counts", {}).items():
            for name, v in counts.items():
                action_mean[name] += v / len(seed_summaries)
    out["action_counts_mean"] = {k: round(v, 1) for k, v in action_mean.items()}
    return out
