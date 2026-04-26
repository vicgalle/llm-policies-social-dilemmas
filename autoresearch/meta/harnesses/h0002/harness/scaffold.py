"""
M2 seed scaffold for production_economy.

Contract (from pipeline/harness.py:M2Scaffold):

    def make_policy(base_model: str | None) -> Callable[[env, int], int]:
        ...

The factory returns a closure ``policy(env, agent_id) -> int`` that the
meta-eval loop calls per agent per step. The scaffold owns any
per-episode mutable state (phase counter, role table, cached LLM
decisions) — the closure captures it via free variables.

Architecture
------------

The scaffold separates *decisions* (high-level choices about what an
agent should be doing) from *mechanics* (low-level navigation, gather,
craft). Decisions are isolated in named hooks marked
``# LLM HOOK:`` — the proposer should turn one or more of these into
``base_model``-gated callouts to wire an LLM into the inference loop.

Decision hooks in this seed (all currently hardcoded):

  1. ``_decide_phase(env, t)`` — what's the macro phase right now?
     Seed: step-thresholded ("early" / "mid" / "late").
  2. ``_decide_role(env, agent_id, phase)`` — what role does this agent
     play in this phase?
     Seed: agent_id parity → wood vs stone.
  3. ``_decide_target_node(env, agent_id, role, phase)`` — which kind
     of cell should the agent head to?
     Seed: deterministic by role + inventory state.

Mechanics (all in code, not decision hooks):

  - ``_bfs_to(env, start, target_set, occupied)`` — agent-aware BFS
  - ``_step_toward(env, agent_id, target_set)`` — wraps BFS into a move
  - ``_dispatch_action(env, agent_id, ...)`` — picks the action given
    a target type and current state
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Action constants (production_economy)
# ---------------------------------------------------------------------------

NOOP = 0
MOVE_N, MOVE_S, MOVE_E, MOVE_W = 1, 2, 3, 4
GATHER = 5
CRAFT = 6
CRAFT_TOOL = 7
CRAFT_SHELTER = 8
DROP_WOOD, DROP_STONE, DROP_PLANK, DROP_BRICK = 9, 10, 11, 12
PICKUP_WOOD, PICKUP_STONE, PICKUP_PLANK, PICKUP_BRICK = 13, 14, 15, 16

W_I, S_I, P_I, B_I = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------


def _bfs_to(env, start, target_set, occupied):
    """Agent-aware BFS. Returns (dr, dc) of first step or None."""
    if not target_set or start in target_set:
        return None
    visited = {start}
    q = deque()
    DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for dr, dc in DIRS:
        nr, nc = start[0] + dr, start[1] + dc
        if not (0 <= nr < env.height and 0 <= nc < env.width):
            continue
        if env.walls[nr, nc]:
            continue
        cell = (nr, nc)
        if cell in occupied and cell not in target_set:
            continue
        if cell in visited:
            continue
        visited.add(cell)
        if cell in target_set:
            return (dr, dc)
        q.append((nr, nc, dr, dc))
    while q:
        r0, c0, fdr, fdc = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r0 + dr, c0 + dc
            if not (0 <= nr < env.height and 0 <= nc < env.width):
                continue
            if env.walls[nr, nc]:
                continue
            cell = (nr, nc)
            if cell in occupied and cell not in target_set:
                continue
            if cell in visited:
                continue
            visited.add(cell)
            if cell in target_set:
                return (fdr, fdc)
            q.append((nr, nc, fdr, fdc))
    return None


def _move_action(dr, dc):
    if dr == -1 and dc == 0: return MOVE_N
    if dr == 1 and dc == 0: return MOVE_S
    if dr == 0 and dc == 1: return MOVE_E
    if dr == 0 and dc == -1: return MOVE_W
    return NOOP


def _step_toward(env, agent_id, target_set, occupied):
    pos = (int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1]))
    res = _bfs_to(env, pos, target_set, occupied)
    if res is None:
        return NOOP
    return _move_action(*res)


# ---------------------------------------------------------------------------
# Decision hooks (LLM hooks live here)
# ---------------------------------------------------------------------------


def _decide_phase(env, t: int) -> str:
    """LLM HOOK 1: phase decision.

    Seed implementation: step-thresholded. Replace with a base_model
    call (e.g., ``base_model.classify(env_summary, options=['early', 'mid', 'late'])``)
    when the proposer wants the LLM to drive macro strategy.
    """
    if t < 80:
        return "early"
    if t < 200:
        return "mid"
    return "late"


def _decide_role(env, agent_id: int, phase: str) -> str:
    """LLM HOOK 2: role assignment.

    Seed implementation: agent-id parity. Replace with a base_model
    call to assign roles based on agent state (current inventory,
    proximity to resources, peers' roles, etc.).
    """
    return "wood" if agent_id % 2 == 0 else "stone"


def _decide_target_node(env, agent_id: int, role: str, phase: str) -> str:
    """LLM HOOK 3: target-node selection.

    Seed implementation: deterministic from role + inventory. Replace
    with a base_model call when the proposer wants finer-grained
    state-conditional dispatch.
    """
    inv = env.inventory[agent_id]
    iw, is_, ip, ib = int(inv[W_I]), int(inv[S_I]), int(inv[P_I]), int(inv[B_I])
    has_tool = bool(env.has_tool[agent_id])

    if role == "wood":
        if iw >= 1:
            return "sawmill"
        if ip >= 2 and ib >= 1 and not has_tool:
            return "forge"
        return "forest"
    else:  # stone
        if is_ >= 1:
            return "masonry"
        if ip >= 2 and ib >= 1 and not has_tool:
            return "forge"
        return "quarry"


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------


def _stocked_targets(env, kind: str):
    """Return positions of stocked resource cells of the given kind."""
    out = set()
    target_type = 0 if kind == "forest" else 1  # FOREST=0, QUARRY=1
    for idx in range(env.n_resources):
        if env.resource_stocked[idx]:
            if int(env.resource_type[idx]) == target_type:
                out.add((int(env.resource_pos[idx, 0]),
                         int(env.resource_pos[idx, 1])))
    return out


def _target_set(env, kind: str):
    if kind == "forest":
        return _stocked_targets(env, "forest")
    if kind == "quarry":
        return _stocked_targets(env, "quarry")
    if kind == "sawmill":
        return env.sawmill_cells_set
    if kind == "masonry":
        return env.masonry_cells_set
    if kind == "forge":
        return env.forge_cells_set
    return set()


def _occupants(env, agent_id: int):
    return {(int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1]))
            for j in range(env.n_agents) if j != agent_id}


def _dispatch_action(env, agent_id: int, target_kind: str) -> int:
    pos = (int(env.agent_pos[agent_id, 0]), int(env.agent_pos[agent_id, 1]))
    inv = env.inventory[agent_id]
    iw, is_, ip, ib = int(inv[W_I]), int(inv[S_I]), int(inv[P_I]), int(inv[B_I])
    has_tool = bool(env.has_tool[agent_id])
    occupied = _occupants(env, agent_id)
    targets = _target_set(env, target_kind)

    if pos in targets:
        if target_kind in ("forest", "quarry"):
            return GATHER
        if target_kind == "sawmill" and iw >= 1:
            return CRAFT
        if target_kind == "masonry" and is_ >= 1:
            return CRAFT
        if target_kind == "forge" and ip >= 2 and ib >= 1 and not has_tool:
            return CRAFT_TOOL
        return NOOP

    # Adjacency check for craft (workshops accept adjacent agents).
    if target_kind in ("sawmill", "masonry"):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (pos[0] + dr, pos[1] + dc) in targets:
                if target_kind == "sawmill" and iw >= 1:
                    return CRAFT
                if target_kind == "masonry" and is_ >= 1:
                    return CRAFT
                break

    return _step_toward(env, agent_id, targets, occupied)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_policy(base_model: Optional[str]) -> Callable:
    """Factory used by pipeline/harness.py:M2Scaffold.

    The seed's closure ignores ``base_model`` (no inference-time LLM
    calls). Proposer mutations should consult ``base_model`` inside
    the decision hooks above.
    """

    def policy(env, agent_id: int) -> int:
        t = int(getattr(env, "_step_count", 0))
        phase = _decide_phase(env, t)
        role = _decide_role(env, agent_id, phase)
        target_kind = _decide_target_node(env, agent_id, role, phase)
        return _dispatch_action(env, agent_id, target_kind)

    # Stash the base_model on the closure so the proposer's instrumented
    # hooks can read it without a global.
    policy.base_model = base_model
    return policy
