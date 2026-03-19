"""
Tabular Q-Learning for the Cleanup Environment
===============================================

Feature-engineered state representation with a shared Q-table across all
agents.  The objective is to maximise collective (team) reward, which
requires agents to learn the clean-vs-collect trade-off: cleaning is
costly (-1) but enables apple spawning for everyone.

State representation (8 discrete features, 11 664 unique states):
  1. BFS action toward nearest apple    (6 vals: F/B/L/R/HERE/NONE)
  2. Distance to nearest apple           (3 bins: 0-3, 4-8, 9+/none)
  3. BFS action toward nearest waste     (6 vals: F/B/L/R/HERE/NONE)
  4. Distance to nearest waste           (3 bins: 0-3, 4-8, 9+/none)
  5. Current river waste density         (3 bins: low/med/high)
  6. Can clean beam hit waste now?       (2 vals: yes/no)
  7. Local alive-apple count             (2 bins: 0, 1+)
  8. Distance to nearest other agent     (3 bins: 1-3, 4-7, 8+/none)

Reward shaping:
  shaped_i = w_ind * r_i
           + w_team * mean(r_all)
           + clean_bonus        (if CLEAN and would hit waste)
           - beam_pen           (if BEAM — almost never beneficial)

Usage:
  python cleanup_qlearning.py
"""

from __future__ import annotations

import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from gathering_env import Orientation, _ROTATIONS
from gathering_policy import direction_to_action
from cleanup_env import (
    CleanupEnv,
    make_cleanup,
    CleanupAction,
    NUM_CLEANUP_ACTIONS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════

# Discretisation boundaries  (value <= bound → that bin index)
_APPLE_DIST_BOUNDS = (3, 8)       # 3 bins: 0-3 | 4-8 | 9+
_WASTE_DIST_BOUNDS = (3, 8)       # 3 bins: 0-3 | 4-8 | 9+
_AGENT_DIST_BOUNDS = (3, 7)       # 3 bins: 1-3 | 4-7 | 8+
_LOCAL_RADIUS      = 5
_LOCAL_APPLE_BOUNDS = (0,)        # 2 bins: 0 | 1+

# Waste density thresholds  (env.threshold_depletion = 0.4 by default)
_WASTE_LOW  = 0.1   # below → healthy, apples spawn well
_WASTE_HIGH = 0.3   # above → danger zone, apples barely spawn


def _bin(value: int, bounds: tuple) -> int:
    for i, b in enumerate(bounds):
        if value <= b:
            return i
    return len(bounds)


def _fbin(value: float, thresholds: tuple) -> int:
    """Bin a float by thresholds."""
    for i, t in enumerate(thresholds):
        if value <= t:
            return i
    return len(thresholds)


# ── BFS to nearest alive apple ──────────────────────────────────────────

def _bfs_apple_info(
    env: CleanupEnv, agent_id: int,
) -> Optional[Tuple[int, int, int]]:
    """Return ``(first_dr, first_dc, bfs_distance)`` or *None*."""
    ar = int(env.agent_pos[agent_id, 0])
    ac = int(env.agent_pos[agent_id, 1])
    start = (ar, ac)

    apple_set: set = set()
    for idx in range(env.n_apples):
        if env.apple_alive[idx]:
            apple_set.add((int(env._apple_pos[idx, 0]),
                           int(env._apple_pos[idx, 1])))
    if not apple_set:
        return None
    if start in apple_set:
        return (0, 0, 0)

    visited = {start}
    queue: deque = deque()
    DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for dr, dc in DIRS:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
            pos = (nr, nc)
            if pos not in visited:
                visited.add(pos)
                if pos in apple_set:
                    return (dr, dc, 1)
                queue.append((nr, nc, dr, dc, 1))

    while queue:
        r, c, fdr, fdc, dist = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
                pos = (nr, nc)
                if pos not in visited:
                    visited.add(pos)
                    if pos in apple_set:
                        return (fdr, fdc, dist + 1)
                    queue.append((nr, nc, fdr, fdc, dist + 1))
    return None


# ── BFS to nearest waste cell ───────────────────────────────────────────

def _bfs_waste_info(
    env: CleanupEnv, agent_id: int,
) -> Optional[Tuple[int, int, int]]:
    """Return ``(first_dr, first_dc, bfs_distance)`` to nearest polluted
    river cell, or *None* if no waste exists."""
    ar = int(env.agent_pos[agent_id, 0])
    ac = int(env.agent_pos[agent_id, 1])
    start = (ar, ac)

    waste_set: set = set()
    for r, c in env.river_cells_list:
        if env.waste[r, c]:
            waste_set.add((r, c))
    if not waste_set:
        return None
    if start in waste_set:
        return (0, 0, 0)

    visited = {start}
    queue: deque = deque()
    DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for dr, dc in DIRS:
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
            pos = (nr, nc)
            if pos not in visited:
                visited.add(pos)
                if pos in waste_set:
                    return (dr, dc, 1)
                queue.append((nr, nc, dr, dc, 1))

    while queue:
        r, c, fdr, fdc, dist = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.height and 0 <= nc < env.width and not env.walls[nr, nc]:
                pos = (nr, nc)
                if pos not in visited:
                    visited.add(pos)
                    if pos in waste_set:
                        return (fdr, fdc, dist + 1)
                    queue.append((nr, nc, fdr, fdc, dist + 1))
    return None


# ── Local apple count ───────────────────────────────────────────────────

def _count_local_apples(env: CleanupEnv, agent_id: int) -> int:
    ar = int(env.agent_pos[agent_id, 0])
    ac = int(env.agent_pos[agent_id, 1])
    count = 0
    for idx in range(env.n_apples):
        if env.apple_alive[idx]:
            pr = int(env._apple_pos[idx, 0])
            pc = int(env._apple_pos[idx, 1])
            if abs(pr - ar) + abs(pc - ac) <= _LOCAL_RADIUS:
                count += 1
    return count


# ── Nearest other active agent ──────────────────────────────────────────

def _nearest_agent_dist(env: CleanupEnv, agent_id: int) -> int:
    """Manhattan distance to nearest active agent, or 999 if none."""
    ar = int(env.agent_pos[agent_id, 0])
    ac = int(env.agent_pos[agent_id, 1])
    best = 999
    for j in range(env.n_agents):
        if j == agent_id or env.agent_timeout[j] > 0:
            continue
        d = abs(int(env.agent_pos[j, 0]) - ar) + abs(int(env.agent_pos[j, 1]) - ac)
        if d < best:
            best = d
    return best


# ── Can the clean beam hit waste right now? ─────────────────────────────

def _can_clean_waste(env: CleanupEnv, agent_id: int) -> bool:
    orient = Orientation(env.agent_orient[agent_id])
    a, b, c, d = _ROTATIONS[orient]
    ar = int(env.agent_pos[agent_id, 0])
    ac = int(env.agent_pos[agent_id, 1])
    half_w = env.beam_width // 2

    for w_off in range(-half_w, half_w + 1):
        for dist in range(1, env.beam_length + 1):
            br = ar + a * dist + b * w_off
            bc = ac + c * dist + d * w_off
            if not (0 <= br < env.height and 0 <= bc < env.width):
                break
            if env.walls[br, bc]:
                break
            if env.waste[br, bc]:
                return True
    return False


# ── Combined state extraction ───────────────────────────────────────────

# Index of the can_clean_now feature inside the state tuple (for reward shaping)
_IDX_CAN_CLEAN = 5

def extract_state(env: CleanupEnv, agent_id: int) -> Tuple[int, ...]:
    """Build an 8-int discretised state tuple."""

    # 1-2: apple navigation
    a_info = _bfs_apple_info(env, agent_id)
    if a_info is None:
        bfs_apple = 5;  apple_dist = 99
    elif a_info[0] == 0 and a_info[1] == 0:
        bfs_apple = 4;  apple_dist = 0
    else:
        bfs_apple = int(direction_to_action(
            a_info[0], a_info[1], int(env.agent_orient[agent_id])))
        apple_dist = a_info[2]
    apple_dist_bin = _bin(apple_dist, _APPLE_DIST_BOUNDS)

    # 3-4: waste navigation
    w_info = _bfs_waste_info(env, agent_id)
    if w_info is None:
        bfs_waste = 5;  waste_dist = 99
    elif w_info[0] == 0 and w_info[1] == 0:
        bfs_waste = 4;  waste_dist = 0
    else:
        bfs_waste = int(direction_to_action(
            w_info[0], w_info[1], int(env.agent_orient[agent_id])))
        waste_dist = w_info[2]
    waste_dist_bin = _bin(waste_dist, _WASTE_DIST_BOUNDS)

    # 5: waste density (global)
    wd = env._compute_waste_density()
    waste_density_bin = _fbin(wd, (_WASTE_LOW, _WASTE_HIGH))

    # 6: can clean now
    can_clean = int(_can_clean_waste(env, agent_id))

    # 7: local apple count
    local_apple_bin = _bin(_count_local_apples(env, agent_id),
                           _LOCAL_APPLE_BOUNDS)

    # 8: nearest agent distance
    agent_dist_bin = _bin(_nearest_agent_dist(env, agent_id),
                          _AGENT_DIST_BOUNDS)

    return (bfs_apple, apple_dist_bin,
            bfs_waste, waste_dist_bin,
            waste_density_bin, can_clean,
            local_apple_bin, agent_dist_bin)


STATE_SPACE_SIZE = 6 * 3 * 6 * 3 * 3 * 2 * 2 * 3   # 11 664


# ═══════════════════════════════════════════════════════════════════════════
# Tabular Q-Learner  (9 actions)
# ═══════════════════════════════════════════════════════════════════════════

class TabularQLearner:
    """Shared Q-table with epsilon-greedy exploration + linear decay."""

    def __init__(
        self,
        n_actions: int = NUM_CLEANUP_ACTIONS,
        alpha: float = 0.1,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 4_000,
        seed: int = 0,
    ):
        self.n_actions = n_actions
        self.q: Dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay_episodes
        self.rng = np.random.default_rng(seed)

    def get_epsilon(self, episode: int) -> float:
        frac = min(episode / max(self.eps_decay, 1), 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def select_action(
        self, state: tuple, episode: int = 0, greedy: bool = False,
    ) -> int:
        if not greedy and self.rng.random() < self.get_epsilon(episode):
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q[state]))

    def update(
        self, state: tuple, action: int,
        reward: float, next_state: tuple, done: bool,
    ) -> None:
        target = reward
        if not done:
            target += self.gamma * float(np.max(self.q[next_state]))
        self.q[state][action] += self.alpha * (target - self.q[state][action])

    @property
    def n_visited(self) -> int:
        return len(self.q)


# ═══════════════════════════════════════════════════════════════════════════
# Reward shaping
# ═══════════════════════════════════════════════════════════════════════════

def shape_rewards(
    rewards: Dict[int, float],
    actions: Dict[int, int],
    states: Dict[int, tuple],
    n_agents: int,
    w_ind: float        = 0.5,
    w_team: float       = 0.5,
    clean_bonus: float  = 1.5,
    beam_pen: float     = 2.0,
) -> Dict[int, float]:
    """Cooperative reward with cleaning incentive.

    - Productive cleaning (action=CLEAN and waste in beam path) earns
      *clean_bonus* on top of the raw -1 cost, making it net-positive.
    - Fire beam gets an extra *beam_pen* penalty (on top of raw -1/-50).
    """
    team_mean = sum(rewards.values()) / n_agents
    shaped: Dict[int, float] = {}
    for i in range(n_agents):
        r = w_ind * rewards[i] + w_team * team_mean
        if actions.get(i) == int(CleanupAction.CLEAN):
            if states[i][_IDX_CAN_CLEAN]:
                r += clean_bonus
        if actions.get(i) == int(CleanupAction.BEAM):
            r -= beam_pen
        shaped[i] = r
    return shaped


# ═══════════════════════════════════════════════════════════════════════════
# Programmatic baselines
# ═══════════════════════════════════════════════════════════════════════════

def _random_policy(env: CleanupEnv, agent_id: int) -> int:
    return int(env.rng.integers(NUM_CLEANUP_ACTIONS))


def _collector_policy(env: CleanupEnv, agent_id: int) -> int:
    """BFS to nearest apple. Never cleans or beams."""
    info = _bfs_apple_info(env, agent_id)
    if info is None:
        return int(CleanupAction.STAND)
    if info[0] == 0 and info[1] == 0:
        return int(CleanupAction.STAND)
    return int(direction_to_action(info[0], info[1],
                                   int(env.agent_orient[agent_id])))


def _cleaner_policy(env: CleanupEnv, agent_id: int) -> int:
    """Priority: clean waste if possible, else navigate toward waste,
    else collect apples if river is already clean."""
    if _can_clean_waste(env, agent_id):
        return int(CleanupAction.CLEAN)
    w_info = _bfs_waste_info(env, agent_id)
    if w_info is not None:
        if w_info[0] == 0 and w_info[1] == 0:
            # Standing on waste but not facing it — try rotating
            return int(CleanupAction.ROTATE_LEFT)
        return int(direction_to_action(w_info[0], w_info[1],
                                       int(env.agent_orient[agent_id])))
    # No waste left — collect apples
    return _collector_policy(env, agent_id)


def _threshold_policy(env: CleanupEnv, agent_id: int) -> int:
    """Clean if waste density > 0.15, else collect apples."""
    if env._compute_waste_density() > 0.15:
        return _cleaner_policy(env, agent_id)
    return _collector_policy(env, agent_id)


# ═══════════════════════════════════════════════════════════════════════════
# Episode runner (Cleanup-specific)
# ═══════════════════════════════════════════════════════════════════════════

def run_cleanup_episode(
    env: CleanupEnv,
    agent_fns: Dict[int, object],
    seed: int = 42,
    verbose: bool = True,
    label: str = "",
) -> dict:
    """Run one full episode; *agent_fns* maps agent id to callable or
    string shortcut: ``"random"``, ``"collector"``, ``"cleaner"``,
    ``"threshold"``."""

    _SHORTCUTS = {
        "random":    _random_policy,
        "collector": _collector_policy,
        "cleaner":   _cleaner_policy,
        "threshold": _threshold_policy,
    }

    fn_map: Dict[int, object] = {}
    tag_map: Dict[int, str] = {}
    for i in range(env.n_agents):
        spec = agent_fns.get(i, "random")
        if isinstance(spec, str) and spec in _SHORTCUTS:
            fn_map[i] = _SHORTCUTS[spec]
            tag_map[i] = spec
        elif callable(spec):
            fn_map[i] = spec
            tag_map[i] = getattr(spec, "__name__", "custom")
        else:
            fn_map[i] = _random_policy
            tag_map[i] = "random"

    env.reset(seed=seed)

    ep_rewards:  Dict[int, List[float]] = {i: [] for i in range(env.n_agents)}
    ep_timeouts: Dict[int, List[bool]]  = {i: [] for i in range(env.n_agents)}
    waste_hist: list[float] = []
    clean_count = 0

    for _step in range(env.max_steps):
        actions: Dict[int, int] = {}
        for i in range(env.n_agents):
            if env.agent_timeout[i] > 0:
                actions[i] = int(CleanupAction.STAND)
            else:
                actions[i] = fn_map[i](env, i)
            if actions[i] == int(CleanupAction.CLEAN):
                clean_count += 1

        _obs, rewards, terminated, _trunc, info = env.step(actions)

        for i in range(env.n_agents):
            ep_rewards[i].append(rewards[i])
            ep_timeouts[i].append(info[i]["timeout"] > 0)
        waste_hist.append(info[0]["waste_density"])

    total = {i: sum(ep_rewards[i]) for i in range(env.n_agents)}
    metrics = CleanupEnv.compute_metrics(ep_rewards, ep_timeouts)
    avg_waste = float(np.mean(waste_hist))

    if verbose:
        if not label:
            label = " + ".join(sorted(set(tag_map.values())))
        print(f"  {label:20s}  team_r={sum(total.values()):7.1f}  "
              f"avg_waste={avg_waste:.3f}  cleans={clean_count}")

    return {
        "total_rewards": total,
        "metrics": metrics,
        "avg_waste": avg_waste,
        "clean_count": clean_count,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(
    n_episodes: int = 5_000,
    n_agents: int   = 3,
    small: bool     = True,
    # Q-learning hypers
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float   = 0.05,
    eps_decay_episodes: int = 4_000,
    # Reward shaping
    w_ind: float       = 0.5,
    w_team: float      = 0.5,
    clean_bonus: float = 1.5,
    beam_pen: float    = 2.0,
    # Misc
    log_every: int = 500,
    seed: int = 42,
) -> TabularQLearner:
    """Train a shared Q-table on the Cleanup environment."""

    env = make_cleanup(n_agents=n_agents, small=small, seed=seed)
    learner = TabularQLearner(
        n_actions=NUM_CLEANUP_ACTIONS,
        alpha=alpha, gamma=gamma,
        eps_start=eps_start, eps_end=eps_end,
        eps_decay_episodes=eps_decay_episodes, seed=seed,
    )

    reward_log: list[float] = []
    waste_log:  list[float] = []

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        states = {i: extract_state(env, i) for i in range(n_agents)}
        ep_team_reward = 0.0
        ep_waste_sum = 0.0

        for _step in range(env.max_steps):
            active = [env.agent_timeout[i] == 0 for i in range(n_agents)]

            actions: Dict[int, int] = {}
            for i in range(n_agents):
                if active[i]:
                    actions[i] = learner.select_action(states[i], episode=ep)
                else:
                    actions[i] = int(CleanupAction.STAND)

            _obs, rewards, terminated, _trunc, info = env.step(actions)
            done = terminated[0]

            shaped = shape_rewards(rewards, actions, states, n_agents,
                                   w_ind, w_team, clean_bonus, beam_pen)

            next_states = {i: extract_state(env, i) for i in range(n_agents)}

            for i in range(n_agents):
                if active[i]:
                    learner.update(states[i], actions[i], shaped[i],
                                   next_states[i], done)

            states = next_states
            ep_team_reward += sum(rewards.values())
            ep_waste_sum += info[0]["waste_density"]

        reward_log.append(ep_team_reward)
        waste_log.append(ep_waste_sum / env.max_steps)

        if (ep + 1) % log_every == 0:
            rec_r = reward_log[-log_every:]
            rec_w = waste_log[-log_every:]
            print(f"  ep {ep+1:>5d} | "
                  f"team_r: {np.mean(rec_r):7.1f} | "
                  f"waste: {np.mean(rec_w):.3f} | "
                  f"eps: {learner.get_epsilon(ep):.3f} | "
                  f"states: {learner.n_visited}")

    return learner


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_q_policy_fn(learner: TabularQLearner):
    """Wrap a trained Q-table into a callable for run_cleanup_episode."""
    def _policy(env: CleanupEnv, agent_id: int) -> int:
        if env.agent_timeout[agent_id] > 0:
            return int(CleanupAction.STAND)
        state = extract_state(env, agent_id)
        return learner.select_action(state, greedy=True)
    _policy.__name__ = "Q-learn"
    return _policy


def evaluate_policy(
    label: str,
    env_factory,
    agent_fns: dict,
    n_episodes: int = 20,
    seed: int = 9999,
) -> dict:
    """Run *n_episodes* and return averaged results."""
    all_rewards: list[float] = []
    all_metrics: list[dict]  = []
    all_waste:   list[float] = []
    all_cleans:  list[int]   = []

    for ep in range(n_episodes):
        env = env_factory()
        result = run_cleanup_episode(
            env, agent_fns, seed=seed + ep, verbose=False)
        all_rewards.append(sum(result["total_rewards"].values()))
        all_metrics.append(result["metrics"])
        all_waste.append(result["avg_waste"])
        all_cleans.append(result["clean_count"])

    avg_r = float(np.mean(all_rewards))
    avg_m = {k: float(np.mean([m[k] for m in all_metrics]))
             for k in all_metrics[0]}
    avg_w = float(np.mean(all_waste))
    avg_c = float(np.mean(all_cleans))
    return {"label": label, "avg_team_reward": avg_r,
            "metrics": avg_m, "avg_waste": avg_w, "avg_cleans": avg_c}


def print_results_table(results: list[dict]) -> None:
    header = (f"  {'Policy':<16s} {'Team R':>8s}  {'Waste':>6s}  "
              f"{'Cleans':>6s}  {'Effic':>7s}  {'Equal':>7s}  "
              f"{'Sustain':>7s}  {'Peace':>7s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        m = r["metrics"]
        print(f"  {r['label']:<16s} {r['avg_team_reward']:>8.1f}  "
              f"{r['avg_waste']:>6.3f}  {r['avg_cleans']:>6.0f}  "
              f"{m['efficiency']:>7.3f}  {m['equality']:>7.3f}  "
              f"{m['sustainability']:>7.1f}  {m['peace']:>7.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    N_TRAIN   = 10
    EPISODES  = 1000
    EVAL_EPS  = 5

    print("=" * 72)
    print("  Tabular Q-Learning — Cleanup (cooperative objective)")
    print("=" * 72)
    print(f"  State space : {STATE_SPACE_SIZE:,} states  (8 features)")
    print(f"  Action space: {NUM_CLEANUP_ACTIONS}")
    print(f"  Training    : {EPISODES} episodes, {N_TRAIN} agents, large map")
    print()

    # ── 1. Train ─────────────────────────────────────────────────────────
    t0 = time.time()
    learner = train(
        n_episodes=EPISODES,
        n_agents=N_TRAIN,
        small=False,
        alpha=0.1,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_episodes=200,
        w_ind=0.5,
        w_team=0.5,
        clean_bonus=1.5,
        beam_pen=2.0,
        log_every=10,
        seed=40,
    )
    elapsed = time.time() - t0
    print(f"\n  Training complete: {elapsed:.1f}s, "
          f"{learner.n_visited:,} / {STATE_SPACE_SIZE:,} states visited\n")

    # ── 2. Evaluate on large map ─────────────────────────────────────────
    print("=" * 72)
    print(f"  Evaluation — Small Map, {N_TRAIN} agents")
    print("=" * 72)

    q_fn = make_q_policy_fn(learner)
    factory_s = lambda: make_cleanup(n_agents=N_TRAIN, small=False, seed=0)

    results_s = [
        evaluate_policy("Q-learn",   factory_s,
                        {i: q_fn for i in range(N_TRAIN)}, EVAL_EPS),
        evaluate_policy("Threshold", factory_s,
                        {i: "threshold" for i in range(N_TRAIN)}, EVAL_EPS),
        evaluate_policy("Cleaner",   factory_s,
                        {i: "cleaner" for i in range(N_TRAIN)}, EVAL_EPS),
        evaluate_policy("Collector", factory_s,
                        {i: "collector" for i in range(N_TRAIN)}, EVAL_EPS),
        evaluate_policy("Random",    factory_s,
                        {i: "random" for i in range(N_TRAIN)}, EVAL_EPS),
    ]
    print()
    print_results_table(results_s)


    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)
