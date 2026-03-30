"""
Pipeline configuration — helper functions for the policy namespace.

The researcher can add extra helper functions here that will be available
to the policy LLM's generated code. These are injected into the policy
namespace alongside the base helpers (bfs_nearest_apple, direction_to_action,
etc.) from gathering_policy.py.

The researcher should also update the system prompt (pipeline/prompts.py)
to document any new helpers added here.

Examples of helpers the researcher might add:
- Coordination primitives (zone assignment, role rotation)
- Pathfinding variants (BFS to waste cells, BFS avoiding opponents)
- State analysis functions (count waste, estimate pollution level)
- Communication protocols (implicit signaling via position/orientation)
"""


def get_extra_helpers() -> dict:
    """Return a dict of extra helper functions for the policy namespace.

    Keys are function names (as strings), values are the callable objects.
    These will be available to the policy code as global names.

    The base helpers from gathering_policy.py are always available.
    Game-specific extras (CleanupAction, etc.) are added by the orchestrator.
    This function provides ADDITIONAL helpers on top of both.
    """
    extra = {}

    # --- Add custom helpers below ---

    # Example (uncomment to add):
    # def count_waste(env):
    #     """Count total waste cells in the river."""
    #     return int(env.waste.sum())
    # extra["count_waste"] = count_waste

    return extra
