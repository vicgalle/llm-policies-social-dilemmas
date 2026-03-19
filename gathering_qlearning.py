"""
Tabular Q-Learning for the Gathering Environment
=================================================

Feature-engineered state representation with a shared Q-table across all
agents.  The objective is to maximise collective (team) apple reward,
encouraging cooperative behaviour (spatial separation, no beaming).

State representation (7 discrete features, 4 320 unique states):
  1. BFS action toward nearest apple   (6 vals: F/B/L/R/HERE/NONE)
  2. Distance to nearest apple          (4 bins: 0, 1-3, 4-8, 9+)
  3. Local alive-apple count            (3 bins: 0, 1-3, 4+)
  4. Nearest other agent direction      (5 vals: F/B/L/R/NONE)
  5. Distance to nearest other agent    (3 bins: 1-3, 4-7, 8+/none)
  6. Can beam an opponent right now     (2 vals: yes/no)
  7. Own accumulated beam-hit count     (2 vals: 0, >=1)

Reward shaping:
  shaped_i = w_ind * r_i  +  w_team * mean(r_all)  -  penalty * used_beam

Usage:
  python gathering_qlearning.py
"""

from __future__ import annotations

import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

from gathering_env import (
    GatheringEnv,
    make_gathering,
    make_gathering_large,
    Action,
    Orientation,
    _ROTATIONS,
    NUM_ACTIONS,
)
from gathering_policy import direction_to_action, run_episode


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════

# Discretisation boundaries (value <= bound → bin index)
_APPLE_DIST_BOUNDS = (0, 3, 8)   # 4 bins: ==0 | 1-3 | 4-8 | 9+
_LOCAL_RADIUS      = 5
_LOCAL_APPLE_BOUNDS = (0, 3)     # 3 bins: 0 | 1-3 | 4+
_AGENT_DIST_BOUNDS = (3, 7)     # 3 bins: 1-3 | 4-7 | 8+


def _bin(value: int, bounds: tuple) -> int:
    for i, b in enumerate(bounds):
        if value <= b:
            return i
    return len(bounds)


# ── BFS to nearest apple (returns first-step + distance) ────────────────

def _bfs_apple_info(
    env: GatheringEnv, agent_id: int,
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


# ── Local apple density ─────────────────────────────────────────────────

def _count_local_apples(env: GatheringEnv, agent_id: int) -> int:
    ar = int(env.agent_pos[agent_id, 0])
    ac = int(env.agent_pos[agent_id, 1])
    count = 0
    for idx in range(env.n_apples):
        if env.apple_alive[idx]:
            pr, pc = int(env._apple_pos[idx, 0]), int(env._apple_pos[idx, 1])
            if abs(pr - ar) + abs(pc - ac) <= _LOCAL_RADIUS:
                count += 1
    return count


# ── Nearest other active agent ──────────────────────────────────────────

def _nearest_agent_info(
    env: GatheringEnv, agent_id: int,
) -> Optional[Tuple[int, int]]:
    """Return ``(relative_dir, manhattan_dist)`` or *None*.

    relative_dir: 0=forward 1=backward 2=left 3=right (in agent's frame).
    """
    ar = int(env.agent_pos[agent_id, 0])
    ac = int(env.agent_pos[agent_id, 1])
    orient = Orientation(env.agent_orient[agent_id])
    a, b, c, d = _ROTATIONS[orient]

    best_dist: float = float("inf")
    best_dir: Optional[int] = None

    for j in range(env.n_agents):
        if j == agent_id or env.agent_timeout[j] > 0:
            continue
        jr = int(env.agent_pos[j, 0])
        jc = int(env.agent_pos[j, 1])
        dist = abs(jr - ar) + abs(jc - ac)
        if dist < best_dist:
            best_dist = dist
            dr, dc = jr - ar, jc - ac
            # Project onto local frame  (det = a*d - b*c = -1 always)
            fwd   = b * dc - d * dr
            right = c * dr - a * dc
            if fwd == 0 and right == 0:
                best_dir = 0
            elif abs(fwd) >= abs(right):
                best_dir = 0 if fwd > 0 else 1
            else:
                best_dir = 3 if right > 0 else 2

    if best_dir is None:
        return None
    return (best_dir, int(best_dist))


# ── Beam-path check ─────────────────────────────────────────────────────

def _can_beam_opponent(env: GatheringEnv, agent_id: int) -> bool:
    orient = Orientation(env.agent_orient[agent_id])
    a, b, c, d = _ROTATIONS[orient]
    ar = int(env.agent_pos[agent_id, 0])
    ac = int(env.agent_pos[agent_id, 1])
    half_w = env.beam_width // 2

    opp_set: set = set()
    for j in range(env.n_agents):
        if j == agent_id or env.agent_timeout[j] > 0:
            continue
        opp_set.add((int(env.agent_pos[j, 0]), int(env.agent_pos[j, 1])))
    if not opp_set:
        return False

    for w_off in range(-half_w, half_w + 1):
        for dist in range(1, env.beam_length + 1):
            br = ar + a * dist + b * w_off
            bc = ac + c * dist + d * w_off
            if not (0 <= br < env.height and 0 <= bc < env.width):
                break
            if env.walls[br, bc]:
                break
            if (br, bc) in opp_set:
                return True
    return False


# ── Combined state extraction ───────────────────────────────────────────

def extract_state(env: GatheringEnv, agent_id: int) -> Tuple[int, ...]:
    """Build a 7-int discretised state tuple for tabular Q-learning."""

    # 1-2: BFS action & distance to nearest apple
    info = _bfs_apple_info(env, agent_id)
    if info is None:
        bfs_act    = 5   # NONE
        apple_dist = 99
    elif info[0] == 0 and info[1] == 0:
        bfs_act    = 4   # HERE (standing on apple)
        apple_dist = 0
    else:
        dr, dc, apple_dist = info
        bfs_act = int(direction_to_action(dr, dc, int(env.agent_orient[agent_id])))
        # direction_to_action returns 0-3 (FORWARD..STEP_RIGHT)

    apple_dist_bin = _bin(apple_dist, _APPLE_DIST_BOUNDS)

    # 3: local apple density
    local_bin = _bin(_count_local_apples(env, agent_id), _LOCAL_APPLE_BOUNDS)

    # 4-5: nearest other agent
    ag_info = _nearest_agent_info(env, agent_id)
    if ag_info is None:
        agent_dir      = 4   # NONE
        agent_dist_bin = 2   # "far / absent"
    else:
        agent_dir      = ag_info[0]
        agent_dist_bin = _bin(ag_info[1], _AGENT_DIST_BOUNDS)

    # 6: can beam
    beam_possible = int(_can_beam_opponent(env, agent_id))

    # 7: own accumulated beam hits
    hits_bin = min(int(env.agent_beam_hits[agent_id]), 1)

    return (bfs_act, apple_dist_bin, local_bin,
            agent_dir, agent_dist_bin, beam_possible, hits_bin)


STATE_SPACE_SIZE = 6 * 4 * 3 * 5 * 3 * 2 * 2   # 4 320


# ═══════════════════════════════════════════════════════════════════════════
# Tabular Q-Learner
# ═══════════════════════════════════════════════════════════════════════════

class TabularQLearner:
    """Shared Q-table with epsilon-greedy exploration + linear decay."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 8000,
        seed: int = 0,
    ):
        self.q: Dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
        )
        self.alpha  = alpha
        self.gamma  = gamma
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay_episodes
        self.rng = np.random.default_rng(seed)

    # ── epsilon schedule ────────────────────────────────────────────────

    def get_epsilon(self, episode: int) -> float:
        frac = min(episode / max(self.eps_decay, 1), 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    # ── action selection ────────────────────────────────────────────────

    def select_action(
        self, state: tuple, episode: int = 0, greedy: bool = False,
    ) -> int:
        if not greedy and self.rng.random() < self.get_epsilon(episode):
            return int(self.rng.integers(NUM_ACTIONS))
        return int(np.argmax(self.q[state]))

    # ── Q update ────────────────────────────────────────────────────────

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
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
    n_agents: int,
    w_ind: float  = 0.5,
    w_team: float = 0.5,
    beam_pen: float = 0.1,
) -> Dict[int, float]:
    """Mix of individual + team-average reward, with beam penalty."""
    team_mean = sum(rewards.values()) / n_agents
    shaped: Dict[int, float] = {}
    for i in range(n_agents):
        r = w_ind * rewards[i] + w_team * team_mean
        if actions.get(i) == Action.BEAM:
            r -= beam_pen
        shaped[i] = r
    return shaped


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train(
    n_episodes: int = 5_000,
    n_agents: int   = 2,
    small: bool     = True,
    # Q-learning hypers
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float   = 0.05,
    eps_decay_episodes: int = 4_000,
    # Reward shaping
    w_ind: float   = 0.5,
    w_team: float  = 0.5,
    beam_pen: float = 0.1,
    # Misc
    log_every: int = 500,
    seed: int = 42,
) -> TabularQLearner:
    """Train a shared Q-table on the Gathering environment."""

    if small:
        env = make_gathering(n_agents=n_agents, small=True, seed=seed)
    else:
        env = make_gathering_large(n_agents=n_agents, seed=seed)
    learner = TabularQLearner(
        alpha=alpha, gamma=gamma,
        eps_start=eps_start, eps_end=eps_end,
        eps_decay_episodes=eps_decay_episodes, seed=seed,
    )

    reward_log: list[float] = []

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        states = {i: extract_state(env, i) for i in range(n_agents)}
        ep_team_reward = 0.0

        for _step in range(env.max_steps):
            # Who is active right now (timeout == 0)?
            active = [env.agent_timeout[i] == 0 for i in range(n_agents)]

            # Select actions
            actions: Dict[int, int] = {}
            for i in range(n_agents):
                if active[i]:
                    actions[i] = learner.select_action(states[i], episode=ep)
                else:
                    actions[i] = int(Action.STAND)

            # Environment step
            _obs, rewards, terminated, _trunc, _info = env.step(actions)
            done = terminated[0]

            # Shaped rewards
            shaped = shape_rewards(rewards, actions, n_agents,
                                   w_ind, w_team, beam_pen)

            # Next states
            next_states = {i: extract_state(env, i) for i in range(n_agents)}

            # Q-updates only for agents that chose their own action
            for i in range(n_agents):
                if active[i]:
                    learner.update(states[i], actions[i], shaped[i],
                                   next_states[i], done)

            states = next_states
            ep_team_reward += sum(rewards.values())

        reward_log.append(ep_team_reward)

        if (ep + 1) % log_every == 0:
            recent = reward_log[-log_every:]
            print(f"  ep {ep+1:>5d} | "
                  f"team_r(last {log_every}): {np.mean(recent):7.1f} | "
                  f"eps: {learner.get_epsilon(ep):.3f} | "
                  f"states: {learner.n_visited}")

    return learner


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_q_policy_fn(learner: TabularQLearner):
    """Wrap a trained Q-table into a callable for ``run_episode``."""
    def _policy(env: GatheringEnv, agent_id: int) -> int:
        if env.agent_timeout[agent_id] > 0:
            return int(Action.STAND)
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
    for ep in range(n_episodes):
        env = env_factory()
        result = run_episode(env, agent_fns, seed=seed + ep, verbose=False)
        all_rewards.append(sum(result["total_rewards"].values()))
        all_metrics.append(result["metrics"])

    avg_r = float(np.mean(all_rewards))
    avg_m = {k: float(np.mean([m[k] for m in all_metrics]))
             for k in all_metrics[0]}
    return {"label": label, "avg_team_reward": avg_r, "metrics": avg_m}


def print_results_table(results: list[dict]) -> None:
    """Pretty-print a comparison table."""
    header = f"  {'Policy':<20s} {'Team Reward':>12s}  {'Effic':>7s}  {'Equal':>7s}  {'Sustain':>7s}  {'Peace':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        m = r["metrics"]
        print(f"  {r['label']:<20s} {r['avg_team_reward']:>12.1f}"
              f"  {m['efficiency']:>7.3f}"
              f"  {m['equality']:>7.3f}"
              f"  {m['sustainability']:>7.1f}"
              f"  {m['peace']:>7.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    N_AGENTS_TRAIN = 10
    TRAIN_EPISODES = 1000
    EVAL_EPISODES  = 5

    print("=" * 68)
    print("  Tabular Q-Learning — Gathering (cooperative objective)")
    print("=" * 68)
    print(f"  State space : {STATE_SPACE_SIZE} states  (7 features)")
    print(f"  Action space: {NUM_ACTIONS}")
    print(f"  Training    : {TRAIN_EPISODES} episodes, {N_AGENTS_TRAIN} agents, large map")
    print()

    # ── 1. Train ─────────────────────────────────────────────────────────
    t0 = time.time()
    learner = train(
        n_episodes=TRAIN_EPISODES,
        n_agents=N_AGENTS_TRAIN,
        small=False,
        alpha=0.1,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_episodes=100,
        w_ind=0.5,
        w_team=0.5,
        beam_pen=0.1,
        log_every=500,
        seed=40,
    )
    elapsed = time.time() - t0
    print(f"\n  Training complete: {elapsed:.1f}s, "
          f"{learner.n_visited} / {STATE_SPACE_SIZE} states visited\n")

    # ── 2. Evaluate on small map (2 agents) ──────────────────────────────
    print("=" * 68)
    print("  Evaluation — Map, 10 agents")
    print("=" * 68)

    q_fn = make_q_policy_fn(learner)
    factory_large = lambda: make_gathering_large(n_agents=10, seed=0)

    results_small = [
        evaluate_policy("Q-learn",     factory_large, {i: q_fn     for i in range(2)}, EVAL_EPISODES),
        evaluate_policy("BFS",         factory_large, {i: "bfs"    for i in range(2)}, EVAL_EPISODES),
        evaluate_policy("Cooperative", factory_large, {i: "coop"   for i in range(2)}, EVAL_EPISODES),
        evaluate_policy("Exploit",     factory_large, {i: "exploit" for i in range(2)}, EVAL_EPISODES),
        evaluate_policy("Random",      factory_large, {i: "random" for i in range(2)}, EVAL_EPISODES),
    ]
    print()
    print_results_table(results_small)


    print("\n" + "=" * 68)
    print("  Done.")
    print("=" * 68)
