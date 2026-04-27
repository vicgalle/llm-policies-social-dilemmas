"""Stackelberg outer best-response loop (arimd_plan.md §3.6).

Pseudocode:

    e = identity_patch
    history = []
    for t in 1..T:
        # Red best-responds to current env e.
        red_candidates = [Red.propose(...) for _ in range(M_red)]
        eval each candidate at λ; pick the one with min V_λ (best for Red).
        pi_R^* = best Red.

        # Blue proposes a JSON patch; evaluate the new env vs the same Red set.
        delta = Blue.propose_patch(e, pi_R^*, history, ...)
        e' = apply_patch(merge(e, delta))
        v' = min_{R in {red_candidates, pi_R^*}} V_λ(e', R)

        # Accept if v' improves on current.
        if v' > best_v_for_blue: e = e'

Outputs are written to ``runs/<tag>/`` for every round: blue patch,
Red candidate codes, eval JSON, and a human-readable history log.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from llm_self_play import log

from . import grammar as grammar_mod
from . import cooperators as coop_mod
from . import evaluator as eval_mod
from . import exploiter as red_mod
from . import designer as blue_mod
from . import diagnostics as diag_mod


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass
class ARIMDConfig:
    game: str
    cooperator: str = "default"           # nested_commons: "efficiency" | "maximin"
    lambda_: float = 0.25
    welfare: str = "blue"                 # "blue" | "population"
    n_agents: Optional[int] = None        # default per env if None
    T: int = 4                            # outer Stackelberg rounds
    M_red: int = 4                        # Red candidates per round
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_assignments: int = 1
    blue_model: str = "claude-opus-4-7"
    red_model: str = "claude-sonnet-4-6"
    max_red_retries: int = 3
    output_dir: Optional[str] = None
    tag: str = "arimd"
    rng_seed: int = 12345
    invasion_lambdas: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.25, 0.4]
    )

    def default_n_agents(self) -> int:
        if self.n_agents is not None:
            return int(self.n_agents)
        return {
            "nested_commons": 16,
            "cleanup": 10,
            "production_economy": 8,
        }[self.game]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_env_source(game: str) -> str:
    return red_mod._env_namespace(game)[2]  # reuse exploiter's path lookup


# ---------------------------------------------------------------------------
# Red search — generate M_red candidates, eval each, pick worst-for-Blue
# ---------------------------------------------------------------------------


async def red_search(
    *,
    cfg: ARIMDConfig,
    env_factory: Callable,
    coop: coop_mod.Cooperator,
    grammar_diff: str,
    round_dir: Path,
) -> Dict[str, Any]:
    log(f"\n  ── Red search (M={cfg.M_red}, model={cfg.red_model}) ──")
    candidates: List[red_mod.RedCandidate] = []
    candidate_evals: List[eval_mod.MixedEvalResult] = []
    history_for_red: List[Dict[str, Any]] = []

    for k in range(cfg.M_red):
        diversity_hint = ""
        if k > 0:
            diversity_hint = (
                f"You have submitted {k} attempt(s) above; their measured "
                "Red and Blue rewards are listed.  Try a *qualitatively* "
                "different exploit — e.g., switch from raiding to "
                "free-riding, or vice versa."
            )
        cand = await red_mod.propose_one(
            game=cfg.game,
            blue_cooperator_code=coop.code,
            blue_cooperator_description=coop.description,
            env_factory_for_smoke=env_factory,
            grammar_diff=grammar_diff,
            lambda_=cfg.lambda_,
            history=history_for_red,
            model=cfg.red_model,
            max_retries=cfg.max_red_retries,
            diversity_hint=diversity_hint,
        )
        if cand is None:
            log(f"  [Red] candidate {k} failed all retries — skipping.")
            history_for_red.append({
                "attempt": k, "v_red": float("nan"), "v_blue": float("nan"),
                "note": "synthesizer failed",
            })
            continue
        # Save the code immediately.
        cand_path = round_dir / f"red_cand_{k:02d}.py"
        cand_path.write_text(cand.code)

        # Evaluate this candidate.
        log(f"  [Red] eval candidate {k} on {len(cfg.seeds)} seeds × "
            f"{cfg.n_assignments} assignments")
        res = eval_mod.evaluate_mixed(
            env_factory, coop.fn, cand.fn,
            lambda_=cfg.lambda_, seeds=cfg.seeds,
            n_assignments=cfg.n_assignments, welfare=cfg.welfare,
            rng_seed=cfg.rng_seed + k,
        )
        candidates.append(cand)
        candidate_evals.append(res)
        log(f"  [Red] cand {k}: V_blue={res.v_blue:.2f}  V_red={res.v_red:.2f}  "
            f"V_pop={res.v_pop:.2f}  raids/ep={res.raid_attempts:.1f}")
        history_for_red.append({
            "attempt": k,
            "v_red": res.v_red,
            "v_blue": res.v_blue,
            "v_pop": res.v_pop,
        })

    if not candidates:
        log("  [Red] no candidates produced this round.")
        return {
            "best_idx": None, "candidates": [], "evals": [], "history": history_for_red,
        }

    # Selfish-defector Red (arimd_plan.md §2.4): pick the candidate that
    # maximizes Red's own return, not the one that minimizes Blue's.
    best_idx = int(np.argmax([r.v_red for r in candidate_evals]))
    log(f"  [Red] best candidate: {best_idx}  V_red={candidate_evals[best_idx].v_red:.2f}  "
        f"V_blue={candidate_evals[best_idx].v_blue:.2f}")
    return {
        "best_idx": best_idx,
        "candidates": candidates,
        "evals": candidate_evals,
        "history": history_for_red,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_arimd(cfg: ARIMDConfig) -> Dict[str, Any]:
    grammar = grammar_mod.get_grammar(cfg.game)
    coop = coop_mod.get_cooperator(cfg.game, cfg.cooperator)

    # Output dir
    if cfg.output_dir is None:
        out_root = Path("autoresearch/arimd/runs") / f"{cfg.tag}_{_now_tag()}"
    else:
        out_root = Path(cfg.output_dir)
    _ensure_dir(out_root)

    # Persist config
    cfg_path = out_root / "config.json"
    cfg_path.write_text(json.dumps({
        "game": cfg.game,
        "cooperator": cfg.cooperator,
        "lambda": cfg.lambda_,
        "welfare": cfg.welfare,
        "n_agents": cfg.default_n_agents(),
        "T": cfg.T,
        "M_red": cfg.M_red,
        "seeds": cfg.seeds,
        "n_assignments": cfg.n_assignments,
        "blue_model": cfg.blue_model,
        "red_model": cfg.red_model,
        "tag": cfg.tag,
        "invasion_lambdas": cfg.invasion_lambdas,
    }, indent=2))

    log("=" * 70)
    log(f"  ARIMD start: tag={cfg.tag} game={cfg.game} coop={cfg.cooperator}")
    log(f"  λ={cfg.lambda_:.3f}  T={cfg.T}  M_red={cfg.M_red}  "
        f"seeds={len(cfg.seeds)}  welfare={cfg.welfare!r}")
    log(f"  blue_model={cfg.blue_model}  red_model={cfg.red_model}")
    log(f"  output: {out_root}")
    log("=" * 70)

    env_source = _read_env_source(cfg.game)

    # Cumulative patch starts as identity.
    current_patch = grammar_mod.identity_patch()
    current_kwargs = grammar_mod.apply_patch(grammar, current_patch)
    current_factory = eval_mod.make_env_factory(
        cfg.game,
        eval_mod.merge_env_kwargs(cfg.game, current_kwargs, cfg.default_n_agents()),
    )

    # Establish V_0 baseline (no Red).
    log("\n── V_0 baseline (no Red) ──")
    v0_res = eval_mod.evaluate_mixed(
        current_factory, coop.fn, None, lambda_=0.0, seeds=cfg.seeds,
        n_assignments=cfg.n_assignments, welfare=cfg.welfare,
    )
    log(f"  V_0 v_blue={v0_res.v_blue:.2f}  eff={v0_res.metrics.get('efficiency', 0):.3f}  "
        f"maximin={v0_res.metrics.get('maximin', 0):.2f}")
    (out_root / "v0_baseline.json").write_text(json.dumps({
        "v_blue": v0_res.v_blue,
        "v_pop": v0_res.v_pop,
        "metrics": v0_res.metrics,
        "n_episodes": v0_res.n_episodes,
    }, indent=2))

    history: List[Dict[str, Any]] = []
    best_v_for_blue: Optional[float] = None
    rounds: List[Dict[str, Any]] = []

    last_red_code: Optional[str] = None
    last_red_rationale: Optional[str] = None
    last_red_v_red: Optional[float] = None

    for t in range(cfg.T):
        round_dir = _ensure_dir(out_root / f"round_{t:02d}")
        log("\n" + "─" * 70)
        log(f"  Stackelberg round {t}/{cfg.T - 1}")
        log("─" * 70)

        # Persist the current env kwargs that Red is responding to.
        (round_dir / "env_kwargs.json").write_text(json.dumps({
            **eval_mod.merge_env_kwargs(cfg.game, current_kwargs, cfg.default_n_agents()),
        }, indent=2, default=str))
        current_diff = grammar_mod.patch_diff(grammar, current_patch)
        (round_dir / "blue_patch_current.json").write_text(json.dumps({
            "patch": current_patch,
            "diff": current_diff,
        }, indent=2))

        # ---------------- Red search ----------------
        red_out = await red_search(
            cfg=cfg, env_factory=current_factory, coop=coop,
            grammar_diff=current_diff, round_dir=round_dir,
        )
        if red_out["best_idx"] is None:
            log("  [round] no Red candidates — accepting empty round.")
            rounds.append({"round": t, "skipped": True})
            continue

        best_red = red_out["candidates"][red_out["best_idx"]]
        best_red_eval = red_out["evals"][red_out["best_idx"]]
        v_for_blue_under_red = best_red_eval.primary(cfg.welfare)

        # Save the chosen Red.
        (round_dir / "red_best.py").write_text(best_red.code)
        (round_dir / "red_best.json").write_text(json.dumps({
            "candidate_idx": red_out["best_idx"],
            "v_blue": best_red_eval.v_blue,
            "v_red": best_red_eval.v_red,
            "v_pop": best_red_eval.v_pop,
            "metrics": best_red_eval.metrics,
            "raid_attempts": best_red_eval.raid_attempts,
            "raid_successes": best_red_eval.raid_successes,
            "reasoning": best_red.reasoning,
        }, indent=2))
        last_red_code = best_red.code
        last_red_rationale = best_red.reasoning
        last_red_v_red = best_red_eval.v_red

        # Initialize best_v at first round (post-Red baseline).
        if best_v_for_blue is None:
            best_v_for_blue = v_for_blue_under_red

        # ---------------- Blue proposes ----------------
        blue_history_for_prompt = [
            {
                "round": h["round"],
                "patch_diff": h["patch_diff"],
                "v_blue": h["v_blue_under_red"],
                "v_pop": h["v_pop_under_red"],
                "v_red": h["v_red_under_red"],
                "raid_attempts": h.get("raid_attempts", 0.0),
                "accepted": h.get("accepted", False),
                "red_rationale": h.get("red_rationale", ""),
            }
            for h in history
        ]

        blue_proposal = await blue_mod.propose_patch(
            game=cfg.game,
            env_source=env_source,
            blue_cooperator_code=coop.code,
            blue_cooperator_description=coop.description,
            grammar=grammar,
            current_patch=current_patch,
            current_diff=current_diff,
            lambda_=cfg.lambda_,
            welfare=cfg.welfare,
            history=blue_history_for_prompt,
            last_red_code=last_red_code,
            last_red_rationale=last_red_rationale,
            last_red_v_red=last_red_v_red,
            model=cfg.blue_model,
        )
        (round_dir / "blue_proposal.json").write_text(json.dumps({
            "patch": blue_proposal.patch,
            "rationale": blue_proposal.rationale,
            "parse_ok": blue_proposal.parse_ok,
            "gen_time": blue_proposal.gen_time,
        }, indent=2, default=str))
        (round_dir / "blue_raw.txt").write_text(blue_proposal.raw_text)

        # Compose the candidate cumulative patch and evaluate the new env
        # against ALL existing Red candidates from this round (so Blue can't
        # win by "tilting away" from one specific exploit).
        candidate_patch = grammar_mod.merge_patches(current_patch, blue_proposal.patch)
        candidate_kwargs = grammar_mod.apply_patch(grammar, candidate_patch)
        candidate_factory = eval_mod.make_env_factory(
            cfg.game,
            eval_mod.merge_env_kwargs(cfg.game, candidate_kwargs, cfg.default_n_agents()),
        )

        # Re-evaluate every Red candidate under the new env.
        new_evals = []
        for k, cand in enumerate(red_out["candidates"]):
            new_res = eval_mod.evaluate_mixed(
                candidate_factory, coop.fn, cand.fn,
                lambda_=cfg.lambda_, seeds=cfg.seeds,
                n_assignments=cfg.n_assignments, welfare=cfg.welfare,
                rng_seed=cfg.rng_seed + 100 + k,
            )
            new_evals.append(new_res)
        worst_under_new = min(r.primary(cfg.welfare) for r in new_evals)
        worst_idx = int(np.argmin([r.primary(cfg.welfare) for r in new_evals]))
        log(f"  [Blue] new patch: V_blue (worst Red) = {worst_under_new:.2f}  "
            f"(prev best = {best_v_for_blue:.2f})")

        accepted = worst_under_new > best_v_for_blue
        if accepted:
            best_v_for_blue = worst_under_new
            current_patch = candidate_patch
            current_kwargs = candidate_kwargs
            current_factory = candidate_factory
            log(f"  [Blue] ACCEPTED")
        else:
            log(f"  [Blue] REJECTED (no improvement)")

        round_record = {
            "round": t,
            "patch_diff": grammar_mod.patch_diff(grammar, candidate_patch),
            "blue_rationale": blue_proposal.rationale,
            "v_blue_under_red": best_red_eval.v_blue,    # under PREVIOUS env state
            "v_pop_under_red": best_red_eval.v_pop,
            "v_red_under_red": best_red_eval.v_red,
            "raid_attempts": best_red_eval.raid_attempts,
            "v_blue_after_patch": worst_under_new,
            "accepted": accepted,
            "red_rationale": last_red_rationale or "",
        }
        history.append(round_record)
        rounds.append(round_record)

        # Update history.jsonl on the fly so a Ctrl-C still leaves a usable trail.
        with (out_root / "history.jsonl").open("a") as f:
            f.write(json.dumps(round_record, default=str) + "\n")

    # ---------------- Final invasion-barrier sweep ----------------
    log("\n" + "=" * 70)
    log("  Final invasion-barrier sweep")
    log("=" * 70)
    # Use the best Red discovered across all rounds (search the history's last red_best).
    last_round_red_path = None
    for t in range(cfg.T - 1, -1, -1):
        candidate = out_root / f"round_{t:02d}" / "red_best.py"
        if candidate.exists():
            last_round_red_path = candidate
            break

    final_table: List[Dict[str, float]] = []
    star = float("nan")
    if last_round_red_path is not None:
        # Reload Red against the FINAL env.
        red_code = last_round_red_path.read_text()
        extra_ns, _, _ = red_mod._env_namespace(cfg.game)
        try:
            final_red_fn = red_mod._load_red_policy(red_code, extra_namespace=extra_ns)
            star, final_table = diag_mod.invasion_barrier(
                current_factory, coop.fn, final_red_fn,
                seeds=cfg.seeds, lambdas=cfg.invasion_lambdas,
                welfare=cfg.welfare, n_assignments=cfg.n_assignments,
            )
            log(f"  Final invasion barrier λ* = {star:.3f}")
            for row in final_table:
                log(f"    λ={row['lambda']:.3f}  v_blue={row['v_blue']:.2f}  "
                    f"v_red={row['v_red']:.2f}  v_pop={row['v_pop']:.2f}")
        except Exception as e:
            log(f"  [final] Red reload failed: {e}")
            final_table = []

    # Stability gap across accepted rounds.
    gaps = diag_mod.stability_gap(history)

    summary = {
        "game": cfg.game,
        "cooperator": cfg.cooperator,
        "lambda": cfg.lambda_,
        "welfare": cfg.welfare,
        "T": cfg.T,
        "M_red": cfg.M_red,
        "seeds": cfg.seeds,
        "v0_baseline_v_blue": v0_res.v_blue,
        "v0_baseline_metrics": v0_res.metrics,
        "best_v_for_blue_after_T": best_v_for_blue,
        "final_patch": current_patch,
        "final_patch_diff": grammar_mod.patch_diff(grammar, current_patch),
        "final_invasion_barrier": star,
        "final_invasion_table": final_table,
        "stability_gaps": gaps,
        "rounds": rounds,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    summary_md = _summary_markdown(cfg, summary)
    (out_root / "summary.md").write_text(summary_md)
    log("\n" + "=" * 70)
    log("  ARIMD complete.")
    log("=" * 70)
    log(summary_md)

    return summary


def _summary_markdown(cfg: ARIMDConfig, summary: Dict[str, Any]) -> str:
    lines = [
        f"# ARIMD run summary — {cfg.tag}",
        "",
        f"- **Game**: `{cfg.game}`  (cooperator: `{cfg.cooperator}`)",
        f"- **λ**: {cfg.lambda_:.3f}  (welfare: `{cfg.welfare}`)",
        f"- **T**: {cfg.T}  M_red: {cfg.M_red}  seeds: {len(cfg.seeds)}",
        f"- **V_0 baseline (Blue)**: {summary['v0_baseline_v_blue']:.2f}",
        f"- **Best V_λ (worst-case Red)**: {summary['best_v_for_blue_after_T']}",
        f"- **Final invasion barrier λ\\***: {summary['final_invasion_barrier']}",
        "",
        "## Final patch",
        "",
        "```",
        summary["final_patch_diff"],
        "```",
        "",
        "## Round-by-round",
        "",
    ]
    for r in summary["rounds"]:
        if r.get("skipped"):
            lines.append(f"- Round {r['round']}: skipped")
            continue
        lines.append(
            f"- **Round {r['round']}** {'(✓ accepted)' if r['accepted'] else '(✗ rejected)'}\n"
            f"    - V_blue under Red: {r['v_blue_under_red']:.2f}\n"
            f"    - V_blue after patch: {r['v_blue_after_patch']:.2f}\n"
            f"    - Blue rationale: {r['blue_rationale']}\n"
            f"    - Red rationale: {r['red_rationale']}"
        )
    if summary["stability_gaps"]:
        lines.append("")
        lines.append("## Stability gaps")
        for i, g in enumerate(summary["stability_gaps"]):
            lines.append(f"- gap {i+1}→{i+2}: {g:.3f}")
    return "\n".join(lines)


def run_arimd_sync(cfg: ARIMDConfig) -> Dict[str, Any]:
    return asyncio.run(run_arimd(cfg))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(
        description="Run an ARIMD Stackelberg search over an env design."
    )
    ap.add_argument("--game", choices=["nested_commons", "cleanup", "production_economy"],
                    required=True)
    ap.add_argument("--cooperator", default="default",
                    help="Cooperator name (nested_commons: efficiency|maximin; "
                         "cleanup: default; production_economy: default)")
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.25)
    ap.add_argument("--welfare", choices=["blue", "population"], default="blue")
    ap.add_argument("--n-agents", type=int, default=None)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--M-red", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5,
                    help="Number of eval seeds (uses 0..seeds-1).")
    ap.add_argument("--n-assignments", type=int, default=1)
    ap.add_argument("--blue-model", default="claude-opus-4-7")
    ap.add_argument("--red-model", default="claude-sonnet-4-6")
    ap.add_argument("--tag", default="arimd")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-red-retries", type=int, default=3)
    ap.add_argument("--rng-seed", type=int, default=12345)
    ap.add_argument("--invasion-lambdas", type=str,
                    default="0.0,0.1,0.2,0.25,0.4",
                    help="Comma-separated λ values for the final barrier sweep.")
    ap.add_argument("--smoke", action="store_true",
                    help="Quick smoke run: T=1, M_red=1, 2 seeds.")
    return ap.parse_args()


def main():
    args = _parse_args()
    if args.smoke:
        args.T = 1
        args.M_red = 1
        args.seeds = 2
    seeds = list(range(int(args.seeds)))
    invasion_lambdas = [float(x) for x in args.invasion_lambdas.split(",")]
    cfg = ARIMDConfig(
        game=args.game,
        cooperator=args.cooperator,
        lambda_=args.lambda_,
        welfare=args.welfare,
        n_agents=args.n_agents,
        T=args.T,
        M_red=args.M_red,
        seeds=seeds,
        n_assignments=args.n_assignments,
        blue_model=args.blue_model,
        red_model=args.red_model,
        tag=args.tag,
        output_dir=args.output_dir,
        max_red_retries=args.max_red_retries,
        rng_seed=args.rng_seed,
        invasion_lambdas=invasion_lambdas,
    )
    return run_arimd_sync(cfg)


if __name__ == "__main__":
    main()
