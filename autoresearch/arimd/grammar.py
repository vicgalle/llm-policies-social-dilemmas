"""Bounded edit grammars per environment (E_G in arimd_plan.md §3.2).

Each grammar exposes a small set of numeric knobs (with absolute bounds)
and structural toggles (categorical/bool). Blue's per-turn output is a
JSON patch that we apply via :func:`apply_patch`, producing a kwargs
dict forwarded to the env factory.

Why kwargs and not a config dataclass: ``cleanup_env`` and
``production_economy_env`` accept their parameters as constructor
kwargs directly. ``nested_commons_env`` builds a ``NestedCommonsConfig``
inside its ``make_nested_commons`` factory from forwarded kwargs. So a
single ``patch -> kwargs`` path works for all three.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Knob types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericKnob:
    name: str
    default: float
    lo: float
    hi: float
    description: str

    def clamp(self, v: float) -> float:
        return max(self.lo, min(self.hi, float(v)))


@dataclass(frozen=True)
class ToggleKnob:
    name: str
    default: Any
    options: tuple
    description: str

    def coerce(self, v: Any) -> Any:
        if v in self.options:
            return v
        # Common-case: booleans encoded as strings or ints.
        if isinstance(self.default, bool):
            if isinstance(v, (int, bool)):
                return bool(v)
            if isinstance(v, str) and v.lower() in ("true", "1", "yes", "on"):
                return True
            if isinstance(v, str) and v.lower() in ("false", "0", "no", "off"):
                return False
        return self.default


@dataclass(frozen=True)
class Grammar:
    game: str
    numeric: Dict[str, NumericKnob]
    toggles: Dict[str, ToggleKnob]

    def describe(self) -> str:
        """Human-readable summary for the Blue prompt."""
        lines = [f"Edit grammar for game = {self.game}", "", "## Numeric knobs"]
        for k, n in self.numeric.items():
            lines.append(
                f"- {n.name:34s} default={n.default:<10g} "
                f"range=[{n.lo:g}, {n.hi:g}]   {n.description}"
            )
        lines.append("")
        lines.append("## Toggles")
        for k, t in self.toggles.items():
            opts = ", ".join(repr(o) for o in t.options)
            lines.append(
                f"- {t.name:34s} default={t.default!r:<10}   options={{{opts}}}   {t.description}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-env grammars
# ---------------------------------------------------------------------------


def _nested_commons_grammar() -> Grammar:
    """Edit grammar for nested_commons_env.

    Numeric knobs cover the dynamics that the prior conversation already
    retuned; toggles cover the held-apple mechanic and a couple of
    cheap-to-implement structural switches.
    """
    numeric = {
        # Plaza shared bonus.
        "bonus_value": NumericKnob(
            "bonus_value", 1.0, 0.5, 2.5,
            "magnitude of the plaza shared bonus payout",
        ),
        "bonus_threshold": NumericKnob(
            "bonus_threshold", 0.25, 0.05, 0.6,
            "max w_P at which a plaza pickup pays the shared bonus",
        ),
        "bonus_regrow_threshold": NumericKnob(
            "bonus_regrow_threshold", 0.35, 0.05, 0.8,
            "max w_P at which plaza bonuses regrow",
        ),
        "bonus_regrow_prob": NumericKnob(
            "bonus_regrow_prob", 0.06, 0.01, 0.2,
            "per-cell plaza-bonus regrow probability",
        ),
        # Orchard.
        "apple_regrow_max": NumericKnob(
            "apple_regrow_max", 0.13, 0.02, 0.3,
            "max per-cell orchard regrow probability when w_q=0",
        ),
        "apple_regrow_slope": NumericKnob(
            "apple_regrow_slope", 2.0, 0.5, 4.0,
            "how strongly w_q suppresses orchard regrow",
        ),
        # Quadrant pollution dynamics.
        "wq_growth": NumericKnob(
            "wq_growth", 0.005, 0.001, 0.02,
            "baseline per-step quadrant waste growth",
        ),
        "wq_growth_per_apple": NumericKnob(
            "wq_growth_per_apple", 0.04, 0.005, 0.15,
            "extra w_q per orchard apple harvested",
        ),
        "wq_clean_amount": NumericKnob(
            "wq_clean_amount", 0.025, 0.005, 0.1,
            "w_q reduction per CLEAN at a river-adjacent cell",
        ),
        # Plaza pollution dynamics.
        "wp_growth": NumericKnob(
            "wp_growth", 0.003, 0.0005, 0.01,
            "baseline per-step plaza waste growth",
        ),
        "wp_growth_per_apple": NumericKnob(
            "wp_growth_per_apple", 0.002, 0.0, 0.01,
            "extra w_P per orchard apple harvested (global coupling)",
        ),
        "wp_clean_amount": NumericKnob(
            "wp_clean_amount", 0.025, 0.005, 0.1,
            "w_P reduction per CLEAN inside the plaza",
        ),
        # Action costs.
        "clean_cost": NumericKnob(
            "clean_cost", 1.0, 0.1, 2.5,
            "reward cost paid by a CLEAN action",
        ),
        "raid_cost": NumericKnob(
            "raid_cost", 0.5, 0.0, 2.0,
            "reward cost paid for issuing a (valid) RAID",
        ),
        "raid_success_prob": NumericKnob(
            "raid_success_prob", 0.6, 0.1, 0.95,
            "P(raid steals 1 held apple given valid attempt)",
        ),
        "travel_step_cost": NumericKnob(
            "travel_step_cost", 0.1, 0.0, 0.5,
            "reward cost per TRAVEL queued move",
        ),
        # Held-apple mechanism (third dilemma).
        "held_apple_per_step_reward": NumericKnob(
            "held_apple_per_step_reward", 0.05, 0.0, 0.2,
            "per-step reward per apple in inventory (incentive to hold)",
        ),
        "initial_held": NumericKnob(
            "initial_held", 2.0, 0.0, 3.0,
            "initial held inventory per agent (seeds raid targets)",
        ),
    }
    toggles: Dict[str, ToggleKnob] = {
        # Plaza shared bonus paid only to other plaza occupants (kills the
        # global free-rider; agents must be in the plaza to benefit).
        "plaza_occupant_only_bonus": ToggleKnob(
            "plaza_occupant_only_bonus", False, (True, False),
            "if True, plaza shared bonus is paid only to agents currently in the plaza",
        ),
        # Plaza payout (individual + shared) gated on the collector's clan
        # river being clean: w_q[clan_of_collector] <= bonus_threshold.
        "plaza_local_clean_gate": ToggleKnob(
            "plaza_local_clean_gate", False, (True, False),
            "if True, plaza bonus requires the collector's clan river to be clean (w_q[clan] ≤ bonus_threshold)",
        ),
        # Same-clan retaliation: a successful cross-clan raid queues every
        # clan-mate of the victim (other than the victim) to auto-RAID the
        # raider next step (adjacency-checked at execute time).
        "same_clan_retaliation": ToggleKnob(
            "same_clan_retaliation", False, (True, False),
            "if True, a successful cross-clan raid forces clan-mates of the victim to auto-RAID the raider next step",
        ),
    }
    return Grammar(game="nested_commons", numeric=numeric, toggles=toggles)


def _cleanup_grammar() -> Grammar:
    """Edit grammar for cleanup_env.

    All knobs map directly to ``CleanupEnv`` constructor kwargs.
    """
    numeric = {
        "fire_cost": NumericKnob(
            "fire_cost", 1.0, 0.1, 5.0,
            "reward cost paid for firing the penalty BEAM",
        ),
        "fire_penalty": NumericKnob(
            "fire_penalty", 50.0, 5.0, 200.0,
            "reward penalty applied to a beamed agent",
        ),
        "clean_cost": NumericKnob(
            "clean_cost", 1.0, 0.1, 3.0,
            "reward cost paid for firing the CLEAN beam",
        ),
        "threshold_depletion": NumericKnob(
            "threshold_depletion", 0.4, 0.1, 0.8,
            "waste density at/above which apples stop spawning",
        ),
        "threshold_restoration": NumericKnob(
            "threshold_restoration", 0.0, 0.0, 0.4,
            "waste density at/below which apple-spawn rate is maximal",
        ),
        "waste_spawn_prob": NumericKnob(
            "waste_spawn_prob", 0.5, 0.05, 1.0,
            "per-step probability of spawning a new waste cell",
        ),
        "apple_respawn_prob": NumericKnob(
            "apple_respawn_prob", 0.05, 0.01, 0.25,
            "per-cell apple respawn probability when river is clean",
        ),
        "beam_length": NumericKnob(
            "beam_length", 5, 1, 10,
            "range (in cells) of both fire and clean beams",
        ),
        "beam_width": NumericKnob(
            "beam_width", 3, 1, 7,
            "transverse width of both beams",
        ),
        "hits_to_tag": NumericKnob(
            "hits_to_tag", 1, 1, 4,
            "BEAM hits required to tag out an agent",
        ),
        "timeout_steps": NumericKnob(
            "timeout_steps", 25, 5, 100,
            "duration (steps) a tagged agent is removed",
        ),
    }
    toggles = {
        # CleanupEnv already accepts beam_enabled — disabling it removes
        # tagging entirely, a legitimate structural lever.
        "beam_enabled": ToggleKnob(
            "beam_enabled", True, (True, False),
            "if False, BEAM is disabled (no tagging)",
        ),
    }
    return Grammar(game="cleanup", numeric=numeric, toggles=toggles)


def _production_economy_grammar() -> Grammar:
    """Edit grammar for production_economy_env. All knobs map to constructor kwargs."""
    numeric = {
        # Tools.
        "T_spoil": NumericKnob(
            "T_spoil", 80, 20, 200,
            "steps until an equipped tool spoils",
        ),
        "tool_step_reward": NumericKnob(
            "tool_step_reward", 2.0, 0.5, 5.0,
            "reward per step per surviving equipped tool",
        ),
        "tool_gather_bonus": NumericKnob(
            "tool_gather_bonus", 1, 0, 3,
            "extra units yielded by GATHER when a tool is equipped",
        ),
        # Winter.
        "winter_step": NumericKnob(
            "winter_step", 200, 100, 280,
            "step at which the winter event fires",
        ),
        "winter_threshold": NumericKnob(
            "winter_threshold", 6, 1, 12,
            "min global shelter_count required to pass winter",
        ),
        "winter_reward": NumericKnob(
            "winter_reward", 50.0, 10.0, 150.0,
            "magnitude of winter reward (±)",
        ),
        # Logistics.
        "inventory_capacity": NumericKnob(
            "inventory_capacity", 3, 2, 6,
            "agent inventory slots",
        ),
        "cell_drop_capacity": NumericKnob(
            "cell_drop_capacity", 5, 1, 12,
            "max items that may sit on a single cell",
        ),
        "resource_respawn_prob": NumericKnob(
            "resource_respawn_prob", 0.05, 0.01, 0.25,
            "per-step respawn probability of a depleted resource cell",
        ),
        "initial_stocked_fraction": NumericKnob(
            "initial_stocked_fraction", 0.8, 0.2, 1.0,
            "fraction of resource nodes stocked at reset",
        ),
    }
    toggles: Dict[str, ToggleKnob] = {}
    return Grammar(
        game="production_economy", numeric=numeric, toggles=toggles
    )


_GRAMMARS = {
    "nested_commons": _nested_commons_grammar(),
    "cleanup": _cleanup_grammar(),
    "production_economy": _production_economy_grammar(),
}


def get_grammar(game: str) -> Grammar:
    if game not in _GRAMMARS:
        raise ValueError(f"No ARIMD grammar for game={game!r}")
    return _GRAMMARS[game]


# ---------------------------------------------------------------------------
# Patch handling
# ---------------------------------------------------------------------------


def identity_patch() -> Dict[str, Dict[str, Any]]:
    """Return the no-op patch — a passthrough to all-defaults."""
    return {"numeric": {}, "toggles": {}}


def normalize_patch(patch: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Coerce a (possibly LLM-output) patch into the canonical structure."""
    if patch is None:
        return identity_patch()
    out = {"numeric": {}, "toggles": {}}
    if isinstance(patch, dict):
        if isinstance(patch.get("numeric"), dict):
            out["numeric"] = dict(patch["numeric"])
        if isinstance(patch.get("toggles"), dict):
            out["toggles"] = dict(patch["toggles"])
    return out


def apply_patch(grammar: Grammar, patch: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a patch into env-factory kwargs.

    Unknown keys in the patch are silently dropped; out-of-range numerics
    are clamped; unknown toggle values fall back to the default.
    """
    p = normalize_patch(patch)
    kwargs: Dict[str, Any] = {}
    for name, knob in grammar.numeric.items():
        if name in p["numeric"]:
            try:
                v = float(p["numeric"][name])
            except (TypeError, ValueError):
                continue
            v = knob.clamp(v)
            # Preserve int-typed knobs (timeout_steps etc.) where the default
            # is integral. Avoids passing 25.0 where 25 was expected.
            if isinstance(knob.default, int) and not isinstance(knob.default, bool):
                v = int(round(v))
            kwargs[name] = v
    for name, knob in grammar.toggles.items():
        if name in p["toggles"]:
            kwargs[name] = knob.coerce(p["toggles"][name])
    return kwargs


def patch_diff(grammar: Grammar, patch: Dict[str, Any]) -> str:
    """Pretty diff vs. defaults — for human-readable round logs."""
    p = normalize_patch(patch)
    lines: List[str] = []
    for name, knob in grammar.numeric.items():
        if name in p["numeric"]:
            v = knob.clamp(float(p["numeric"][name]))
            if isinstance(knob.default, int) and not isinstance(knob.default, bool):
                v = int(round(v))
            if v != knob.default:
                lines.append(f"  {name}: {knob.default} -> {v}")
    for name, knob in grammar.toggles.items():
        if name in p["toggles"]:
            v = knob.coerce(p["toggles"][name])
            if v != knob.default:
                lines.append(f"  {name}: {knob.default!r} -> {v!r}")
    if not lines:
        return "  (identity patch — no edits)"
    return "\n".join(lines)


def merge_patches(base: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """Compose two patches: ``delta`` overrides ``base`` field-by-field."""
    b = normalize_patch(base)
    d = normalize_patch(delta)
    out = {
        "numeric": {**b["numeric"], **d["numeric"]},
        "toggles": {**b["toggles"], **d["toggles"]},
    }
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for game in ("nested_commons", "cleanup", "production_economy"):
        g = get_grammar(game)
        print("=" * 60)
        print(g.describe())
        print()
        # Round-trip sanity.
        kwargs = apply_patch(g, identity_patch())
        assert kwargs == {}, f"identity patch should produce empty kwargs, got {kwargs}"
        # Out-of-range clamping.
        any_n = next(iter(g.numeric))
        kwargs = apply_patch(g, {"numeric": {any_n: 1e9}})
        assert kwargs[any_n] <= g.numeric[any_n].hi
    print("grammar.py self-test OK")
