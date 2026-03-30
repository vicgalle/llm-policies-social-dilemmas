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
"""

import textwrap


# Social metric definitions shown to the policy LLM.
METRIC_DEFINITIONS = (
    "- **Efficiency**: collective apple collection rate across all agents "
    "(higher = more apples collected per step).\n"
    "- **Equality**: fairness of reward distribution between agents "
    "(1.0 = perfectly equal, negative = highly unequal).\n"
    "- **Sustainability**: long-term apple availability — measures whether "
    "resources are preserved over the episode (higher = apples remain available "
    "later in the episode).\n"
    "- **Peace**: absence of aggressive beaming — counts agents not involved in "
    "attack beam conflicts (higher = less aggression). Using the CLEAN beam to "
    "remove waste does NOT reduce peace."
)


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

    Default: reward + social metrics (the "dense" feedback of Gallego 2026).

    Args:
        iteration: Current iteration index (0 = from scratch)
        n_total: Total iterations
        opponent: Previous PolicyRecord (None for iteration 0)
        history: List of dicts with keys: iteration, reward_avg, metrics
        env_factory: Callable that creates environment instances
        game_config: GameConfig with game-specific settings
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
