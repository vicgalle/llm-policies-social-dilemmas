"""Blue — LLM-driven env-edit proposer (arimd_plan.md §3.5).

One LLM call per Stackelberg round. Output: a JSON patch over the
bounded grammar (``grammar.py``). Robust JSON extraction and
silent-fallback to identity patch on parse failure.

Blue sees:
- The env source code with the *current accumulated patch* applied
  (string-form so Blue isn't expected to chase imports).
- The frozen Blue cooperator's source.
- The grammar (numeric ranges + toggles).
- A history of (patch, V_λ_blue, V_λ_pop, V_red, raid_stats) per round.
- The latest Red policy and its measured exploit gain.
"""

from __future__ import annotations

import asyncio
import json
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_self_play import _call_llm, log

from . import grammar as grammar_mod


_SYSTEM_PROMPT = textwrap.dedent("""\
You are *Blue*, an environment designer in an adversarially-robust
mechanism design loop.

A bounded grammar of allowed edits is given.  You commit to an edit, then
a strategic *Red* exploiter writes the most damaging policy it can
against your edited environment, with λ-fraction agents playing Red and
the rest playing a fixed cooperator.

You optimize V_λ = (welfare functional) over the **Blue agents** by
default — equivalently, you want to make the Blue cooperator robust to
adversarial invasion.  Concretely: pick the patch that, after Red's best
response, leaves the worst-off-Blue per-agent return as high as possible.

Per arimd_plan.md §6.1, the contribution is *which structural rule edits
you propose*, not pure parameter tuning.  Prefer rule toggles or
qualitatively new dynamics over inch-by-inch numeric scaling.

Output **exactly one** JSON object inside a ```json ... ``` block, with
this schema:

    {
      "numeric": { "<knob_name>": <float>, ... },
      "toggles": { "<toggle_name>": <value>, ... },
      "rationale": "<1-3 sentences on what assumption you're hardening>"
    }

Omit knobs you don't want to change.  Out-of-range numerics are clamped.
Unknown knob names are silently dropped.
""")


_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
_FENCED_BLOCK_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM response, robustly."""
    for matcher in (_JSON_BLOCK_RE, _FENCED_BLOCK_RE):
        for block in matcher.findall(text):
            try:
                obj = json.loads(block.strip())
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    # Last resort: find the largest brace-balanced substring and try it.
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for s in starts:
        depth = 0
        for j in range(s, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    blob = text[s:j + 1]
                    try:
                        obj = json.loads(blob)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
                    break
    return None


def _format_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "(this is the first Blue turn — no prior rounds)"
    lines = []
    for h in history:
        diff = h.get("patch_diff", "(identity)")
        lines.append(
            f"### Round {h['round']}\n"
            f"  patch:\n{textwrap.indent(diff, '    ')}\n"
            f"  V_λ blue = {h.get('v_blue', float('nan')):.2f}   "
            f"V_λ pop = {h.get('v_pop', float('nan')):.2f}   "
            f"V_red = {h.get('v_red', float('nan')):.2f}\n"
            f"  raid_attempts/ep = {h.get('raid_attempts', 0):.1f}   "
            f"accepted = {h.get('accepted', False)}\n"
            f"  Red rationale: {h.get('red_rationale', '(none)')}"
        )
    return "\n\n".join(lines)


def build_blue_prompt(
    *,
    game: str,
    env_source: str,
    blue_cooperator_code: str,
    blue_cooperator_description: str,
    grammar: grammar_mod.Grammar,
    current_patch: Dict[str, Any],
    current_diff: str,
    lambda_: float,
    welfare: str,
    history: List[Dict[str, Any]],
    last_red_code: Optional[str],
    last_red_rationale: Optional[str],
    last_red_v_red: Optional[float],
) -> str:
    parts = [
        f"# Blue turn — {game}",
        "",
        f"λ = {lambda_:.3f}.  Welfare convention: {welfare!r}.",
        "",
        "## Edit grammar (your action space)",
        "",
        grammar.describe(),
        "",
        "## Current patch (cumulative — your starting point)",
        "",
        f"```\n{current_diff}\n```",
        "",
        "## Env source (with the current patch applied at runtime)",
        "",
        "```python",
        env_source,
        "```",
        "",
        "## Frozen Blue cooperator",
        "",
        f"_{blue_cooperator_description}_",
        "",
        "```python",
        blue_cooperator_code,
        "```",
        "",
        "## History",
        "",
        _format_history(history),
        "",
    ]
    if last_red_code is not None:
        parts.extend([
            "## Last Red policy (the exploit you must close)",
            "",
            f"_Red mean reward = {last_red_v_red if last_red_v_red is not None else float('nan'):.2f}_",
            "",
            "```python",
            last_red_code,
            "```",
            "",
            f"Red's stated rationale: {last_red_rationale or '(none)'}",
            "",
        ])
    parts.extend([
        "## Your task",
        "",
        "Propose a patch (JSON) that improves V_λ (Blue welfare) under "
        "Red's best response.  Prefer structural toggles over fine numeric "
        "tuning; explain your move in `rationale`.",
        "",
        "Output ONE ```json ... ``` block with the schema described in the "
        "system prompt.  Edits to your *current* patch are absolute "
        "values, not deltas (e.g. `bonus_value: 0.5` sets it to 0.5, not "
        "shifts by 0.5).",
    ])
    return "\n".join(parts)


@dataclass
class BluePatchProposal:
    patch: Dict[str, Any]
    rationale: str
    raw_text: str
    gen_time: float
    parse_ok: bool


async def propose_patch(
    *,
    game: str,
    env_source: str,
    blue_cooperator_code: str,
    blue_cooperator_description: str,
    grammar: grammar_mod.Grammar,
    current_patch: Dict[str, Any],
    current_diff: str,
    lambda_: float,
    welfare: str,
    history: List[Dict[str, Any]],
    last_red_code: Optional[str],
    last_red_rationale: Optional[str],
    last_red_v_red: Optional[float],
    model: str,
) -> BluePatchProposal:
    user_prompt = build_blue_prompt(
        game=game,
        env_source=env_source,
        blue_cooperator_code=blue_cooperator_code,
        blue_cooperator_description=blue_cooperator_description,
        grammar=grammar,
        current_patch=current_patch,
        current_diff=current_diff,
        lambda_=lambda_,
        welfare=welfare,
        history=history,
        last_red_code=last_red_code,
        last_red_rationale=last_red_rationale,
        last_red_v_red=last_red_v_red,
    )
    log(f"  [Blue] proposing patch ({model})")
    t0 = time.time()
    try:
        text, _reasoning = await _call_llm(_SYSTEM_PROMPT, user_prompt, model)
    except Exception as e:
        log(f"  [Blue] LLM call failed: {e}")
        return BluePatchProposal(
            patch=grammar_mod.identity_patch(),
            rationale=f"(LLM call failed: {e})",
            raw_text="",
            gen_time=time.time() - t0,
            parse_ok=False,
        )

    obj = _extract_json(text)
    if obj is None:
        log("  [Blue] could not parse a JSON patch; falling back to identity")
        return BluePatchProposal(
            patch=grammar_mod.identity_patch(),
            rationale="(parse failed; identity patch used)",
            raw_text=text,
            gen_time=time.time() - t0,
            parse_ok=False,
        )

    rationale = obj.pop("rationale", "")
    patch = grammar_mod.normalize_patch(obj)
    return BluePatchProposal(
        patch=patch,
        rationale=str(rationale),
        raw_text=text,
        gen_time=time.time() - t0,
        parse_ok=True,
    )


def propose_patch_sync(*args, **kwargs) -> BluePatchProposal:
    return asyncio.run(propose_patch(*args, **kwargs))
