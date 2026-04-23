"""
Pipeline configuration — feedback construction.

The researcher can modify this to change what information the policy LLM
receives between iterations. This is the core "feedback engineering" component.

Modifications might include:
- Changing which metrics to show (subset, derived metrics, per-agent stats)
- Reframing the language (cooperative vs competitive, hints vs neutral)
- Adding trajectory analysis (trends, comparisons)
- Adding strategic suggestions based on metrics
- Changing the iteration prompt structure

Profile-Guided Autoresearch (PGA):
  When ``history[i]["profile_markdown"]`` is non-empty, this builder embeds
  the execution profile (action histograms, change-points, inter-agent
  divergence, branch coverage, precondition-failure rates) after the metric
  block so the policy LLM can target specific mechanisms to mutate. See
  ``pipeline/profile.py`` for the channel definitions.
"""

import textwrap


# Social metric definitions shown to the policy LLM.
METRIC_DEFINITIONS = (
    "- **Efficiency**: collective reward rate per step across all agents "
    "(higher = more total reward earned per timestep).\n"
    "- **Equality**: fairness of reward distribution between agents "
    "(1.0 = perfectly equal, lower = more unequal).\n"
    "- **Sustainability**: the mean timestep at which reward-producing events "
    "occur (higher = rewards accrue later in the episode, indicating preserved "
    "resources or delayed payoffs).\n"
    "- **Peace**: absence of aggressive interaction (higher = less conflict). "
    "Games without a tagging/attack mechanism always report the maximum."
)


def _profile_hint_lines(profile: dict) -> list[str]:
    """Derive a short, action-oriented "what to try next" from the profile.

    Mirrors the diagnostic style recommended by the PGA plan: the LLM should
    attribute observed outcomes to specific mechanisms (cold branches, unmet
    preconditions, no specialization) rather than guess blindly.
    """
    hints: list[str] = []
    never = profile.get("actions_never_used") or []
    if never:
        hints.append(
            f"- The following actions were **never invoked** across the whole "
            f"episode: {', '.join(never)}. If any of them are load-bearing "
            "for the game's reward structure, consider adding a branch that "
            "triggers them."
        )
    cp_mean = profile.get("change_points_per_agent_mean", 0.0)
    if cp_mean == 0.0:
        hints.append(
            "- Every agent's reward stream is **stationary** (no detected "
            "change points). If the reward structure is not uniform over "
            "time, the policy is not reacting to that structure."
        )
    dead = profile.get("dead_branches") or []
    if dead:
        hints.append(
            "- The current policy contains **dead branches** (lines that "
            "never executed). These are silent bugs or unreachable cases — "
            "prune or fix the guards that prevent them from firing."
        )
    tv = profile.get("action_tv_mean", 0.0)
    r2 = profile.get("role_id_rsquared", 0.0)
    if tv < 0.05:
        hints.append(
            "- Agents are behaving nearly identically in self-play. If the "
            "task benefits from a division of labour, introduce a branch that "
            "conditions on `agent_id` or on a deterministic hash of state."
        )
    elif r2 > 0.5:
        hints.append(
            "- Behaviour is largely predicted by `agent_id` alone (role-R² "
            "is high). This is a **brittle static assignment**; prefer "
            "state-conditional role selection so specialisation adapts to "
            "what resources are available."
        )
    ineff = profile.get("action_ineffective_rate") or {}
    top_bad = [(n, r) for n, r in ineff.items() if r >= 0.5]
    if top_bad:
        parts = ", ".join(f"`{n}` ({int(r*100)}%)" for n, r in top_bad[:3])
        hints.append(
            f"- High ineffective-action rates on: {parts}. These invocations "
            "produced no state change — either the precondition is unmet or "
            "the agent is standing on the wrong cell."
        )
    inv_full = profile.get("inventory_full_frame_rate", 0.0)
    if inv_full >= 0.3:
        hints.append(
            f"- Agents spend {int(inv_full*100)}% of frames at inventory "
            "cap: the cap is a binding constraint. Any action that requires "
            "spare inventory capacity will silently no-op while this is true."
        )
    return hints


def build_iteration_prompt(
    iteration: int,
    n_total: int,
    opponent,  # PolicyRecord or None
    history: list[dict],
    env_factory: callable = None,
    game_config=None,
) -> str:
    """Build the user prompt for a policy synthesis iteration.

    This is the main feedback construction function. The researcher can modify
    this to change what information the policy LLM receives.

    Default: reward + social metrics + PGA profile (if available).
    """
    parts = []

    if opponent is None:
        parts.append(f"## Iteration {iteration}/{n_total}: Write the initial policy\n")
        parts.append(
            "No prior policy exists yet. All agents will run the same code. "
            "Your task is to write a first policy that maximizes per-agent reward.\n"
        )
    else:
        parts.append(f"## Iteration {iteration}/{n_total}: Write an improved policy\n")
        parts.append(
            "The following policy is currently used by all agents. "
            "All agents run the same code. "
            "Your task is to write an improved version that maximizes per-agent reward.\n"
        )
        parts.append(f"### Current policy: **{opponent.name}**\n")
        parts.append(f"```python\n{opponent.code}\n```\n")

    if history:
        parts.append("## Results from previous iterations\n")
        parts.append("### Social Metrics (definitions)\n")
        parts.append(METRIC_DEFINITIONS)
        parts.append("")
        for h in history:
            m = h["metrics"]
            parts.append(
                f"- Iteration {h['iteration']}: "
                f"Avg agent reward={h['reward_avg']:.1f} | "
                f"efficiency={m['efficiency']:.3f}, "
                f"equality={m.get('equality', 0):.3f}, "
                f"sustainability={m.get('sustainability', 0):.1f}, "
                f"peace={m.get('peace', 0):.1f}"
            )
        parts.append("")

        # --- MH mode: surface accept/reject status of each prior iter ---
        mh_lines = []
        for h in history:
            if "mh_accepted" in h and h["mh_accepted"] is not None:
                status = "ACCEPTED" if h["mh_accepted"] else "REJECTED"
                b = h.get("mh_beta")
                mh_lines.append(
                    f"- Iteration {h['iteration']}: MH {status}"
                    + (f" (β={b:.2f})" if b is not None else "")
                )
        if mh_lines:
            parts.append("### Metropolis–Hastings trace")
            parts.extend(mh_lines)
            if history and history[-1].get("mh_accepted") is False:
                parts.append(
                    "\nThe previous mutation was **rejected** because it did not "
                    "improve the primary metric. The accepted policy above is "
                    "the one to edit; do not repeat the rejected mutation."
                )
            parts.append("")

        # --- PGA: embed profile of the most recent iteration + hints ------
        latest = history[-1]
        profile_md = latest.get("profile_markdown")
        profile_dict = latest.get("profile")
        if profile_md:
            parts.append("## Execution profile of the previous policy\n")
            parts.append(
                "Scalar reward tells you *that* the policy underperforms; the "
                "profile below tells you *where*. Treat it as a diagnostic: "
                "attribute the observed reward to specific mechanisms (cold "
                "branches, unmet preconditions, lack of specialisation) and "
                "propose mutations that target what the profile surfaces.\n"
            )
            parts.append(profile_md)
            parts.append("")
            hints = _profile_hint_lines(profile_dict or {})
            if hints:
                parts.append("### Targeted mutation suggestions\n")
                parts.extend(hints)
                parts.append("")

    # Environment description
    env_desc = _env_description(env_factory) if env_factory else "agents on a map"
    env_hint = game_config.env_hint if game_config else ""
    max_action = game_config.max_action if game_config else 8

    parts.append(textwrap.dedent(f"""\
    ## Instructions

    Write a policy that maximizes per-agent reward. All agents will run your
    exact same code simultaneously. There are {env_desc}.
    {env_hint}

    Write your `policy(env, agent_id) -> int` function (returns 0-{max_action}).
    """))

    return "\n".join(parts)


def _env_description(env_factory: callable) -> str:
    """Generate a short description of the map from an env factory."""
    env = env_factory()
    return (
        f"{env.n_agents} agents on a {env.width}x{env.height} map "
        f"with ~{env.n_apples} apple spawns"
    )
