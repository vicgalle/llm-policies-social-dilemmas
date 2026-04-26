"""Extra helper bindings injected into the synthesized policy's namespace.

Whatever this returns is added to the policy's globals at exec time.
The seed ships an empty dict — the synthesizer LLM has to write any
helpers it needs inline. The proposer can add e.g. a BFS implementation
or a navigation primitive to make the synthesizer's job easier (or
harder, if part of the search question is "do helpers help?").
"""

from __future__ import annotations


def get_extra_helpers() -> dict:
    return {}
