"""Per-iteration feedback construction for the M1 synthesizer.

The seed version shows the previous policy's source + average reward.
No social metrics, no execution profile, no worked diff hints. The
proposer can enrich this freely.
"""

from __future__ import annotations


def _action_count(game: str) -> int:
    return {
        "production_economy": 17,
        "cleanup": 9,
        "gathering": 8,
        "coop_mining": 8,
    }.get(game, 8)


def build_iteration_prompt(
    iteration: int,
    n_total: int,
    opponent,                    # PolicyRecord | None
    history: list,
    env_factory=None,
    game_config=None,
) -> str:
    game = game_config.name if game_config is not None else "production_economy"
    parts: list[str] = []

    if opponent is None:
        parts.append(
            f"## Iteration {iteration}/{n_total} — initial policy\n\n"
            f"No prior policy exists yet. All agents will run the same code "
            f"in self-play (n_agents matches the env's default). Write a "
            f"first policy that earns positive reward.\n"
        )
    else:
        parts.append(
            f"## Iteration {iteration}/{n_total} — improve the prior policy\n\n"
            f"All agents currently run the policy below. Output a new "
            f"version that earns higher average per-agent reward in self-play.\n\n"
            f"### Previous policy ({opponent.name})\n\n"
            f"```python\n{opponent.code}\n```\n"
        )

    if history:
        parts.append("### Reward history (per-agent average)\n")
        for h in history:
            parts.append(f"- iteration {h['iteration']}: "
                         f"reward_avg = {h['reward_avg']:.2f}")
        parts.append("")

    n_actions = _action_count(game)
    parts.append(
        f"Action range: 0 .. {n_actions - 1}. Read the env source for "
        f"the action enum."
    )

    return "\n".join(parts)
