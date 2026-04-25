"""
Proposer toolkit (CLI + Python API).

The proposer (Claude Code) interacts with the meta-harness state via these
subcommands. Each is also importable from Python::

    from autoresearch.meta.tools import list_harnesses, eval_harness, ...

Path-whitelist guard
--------------------

The proposer's writes via ``spawn``, ``probe``, ``write``, and ``log`` are
constrained to:

* ``autoresearch/meta/harnesses/<hid>/`` for the *new* harness id only;
* ``autoresearch/meta/scratch/`` for probes;
* ``autoresearch/meta/proposer_log.md`` for narrative entries.

Existing harnesses are read-only after creation (forces causal reasoning:
to "fix" an idea, the proposer must spawn a new branch). Frozen
infrastructure (``cleanup_env.py``, ``llm_self_play.py``, ``pipeline/`` —
EXCEPT generated harness pipelines —, ``run_inner_loop.py``) cannot be
written via these tools, period. Raw ``Edit``/``Write`` is the proposer's
own responsibility — see meta/program.md.

Subcommands
-----------

::

    list                                # list harness ids + brief metrics
    read <hid>                          # print manifest + rationale + metrics
    eval <hid> [--seeds N] [--objectives ...]
                                        # adaptive eval (writes metrics.json)
    diff <hid_a> <hid_b>                # unified diff between harness sources
    pareto [--objectives ...]           # current Pareto frontier ids
    spawn <parent_hid> [--mode M3] [--rationale TEXT] [--base-model X]
        [--edits-from PATH] [--id NEWID]
                                        # create a new harness branch
    trace <hid> --seed N [--lines K]    # head a per-step trace
    summary <hid>                       # aggregated traces summary
    probe write <name> < CODE           # save proposer-authored probe
    probe run <name> [args...]          # exec probe; capture stdout
    log <markdown>                      # append to proposer_log.md
    next-id                             # print next free hNNNN id

Exit codes: 0 success, 1 user error, 2 internal error.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
META_DIR = REPO_ROOT / "autoresearch" / "meta"
HARNESS_DIR = META_DIR / "harnesses"
SCRATCH_DIR = META_DIR / "scratch"
LOG_PATH = META_DIR / "proposer_log.md"

sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Path whitelist
# ---------------------------------------------------------------------------


def _is_under(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _assert_writable(target: Path, *, allow_existing: bool = False) -> None:
    """Reject writes outside the proposer-writable subtree."""
    target = target.resolve()
    if _is_under(target, SCRATCH_DIR):
        return
    if target == LOG_PATH.resolve():
        return
    if _is_under(target, HARNESS_DIR):
        # Harnesses are immutable after creation — refuse if the harness's
        # manifest.json already exists and the file is inside that harness.
        try:
            rel = target.relative_to(HARNESS_DIR)
            hid = rel.parts[0]
            manifest = HARNESS_DIR / hid / "manifest.json"
            if manifest.exists() and not allow_existing:
                raise PermissionError(
                    f"harness {hid} is sealed (has manifest.json). "
                    f"To change, spawn a new branch."
                )
            return
        except IndexError:
            pass
    raise PermissionError(
        f"writes to {target} are not allowed. "
        f"Allowed roots: {HARNESS_DIR} (new harnesses), {SCRATCH_DIR}, "
        f"{LOG_PATH}."
    )


# ---------------------------------------------------------------------------
# ID allocation
# ---------------------------------------------------------------------------


def next_id() -> str:
    """Return the next free harness id (h0000, h0001, ...)."""
    used: set[int] = set()
    for d in HARNESS_DIR.glob("h*"):
        m = re.match(r"^h(\d+)$", d.name)
        if m:
            used.add(int(m.group(1)))
    archive = META_DIR / "archive"
    if archive.exists():
        for d in archive.glob("h*"):
            m = re.match(r"^h(\d+)$", d.name)
            if m:
                used.add(int(m.group(1)))
    n = 0
    while n in used:
        n += 1
    return f"h{n:04d}"


# ---------------------------------------------------------------------------
# list / read / pareto / next-id
# ---------------------------------------------------------------------------


def list_harnesses(filter_env: Optional[str] = None,
                   filter_mode: Optional[str] = None) -> List[dict]:
    out = []
    for hd in sorted(HARNESS_DIR.glob("h*")):
        manifest_path = hd / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if filter_env and manifest.get("env") != filter_env:
            continue
        if filter_mode and manifest.get("mode") != filter_mode:
            continue
        metrics = {}
        mp = hd / "metrics.json"
        if mp.exists():
            try:
                m = json.loads(mp.read_text())
                metrics = {
                    "n_seeds": m.get("n_seeds"),
                    "efficiency": (m.get("primary", {}) or {}).get("efficiency"),
                    "maximin": (m.get("primary", {}) or {}).get("maximin"),
                    "stage_reached": m.get("stage_reached"),
                }
            except Exception:
                pass
        out.append({
            "id": manifest.get("id", hd.name),
            "mode": manifest.get("mode"),
            "env": manifest.get("env"),
            "parent": manifest.get("parent"),
            "tag": manifest.get("tag"),
            "metrics": metrics,
        })
    return out


def read_harness(hid: str) -> dict:
    hd = HARNESS_DIR / hid
    if not hd.exists():
        raise FileNotFoundError(hid)
    manifest = json.loads((hd / "manifest.json").read_text())
    rationale = ""
    rp = hd / "rationale.md"
    if rp.exists():
        rationale = rp.read_text()
    metrics = None
    mp = hd / "metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text())
    src = {}
    harness_subdir = hd / "harness"
    if harness_subdir.exists():
        for f in sorted(harness_subdir.rglob("*.py")):
            src[str(f.relative_to(hd))] = f.read_text()
    summary_path = hd / "traces" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return {
        "manifest": manifest,
        "rationale": rationale,
        "metrics": metrics,
        "source": src,
        "trace_summary": summary,
    }


# ---------------------------------------------------------------------------
# eval / trace
# ---------------------------------------------------------------------------


def eval_harness(hid: str, *, seeds: Optional[int] = None,
                 objectives: Optional[List[str]] = None) -> dict:
    from autoresearch.meta.evaluator import adaptive_eval
    force_stage = None
    if seeds is not None:
        if seeds <= 2:
            force_stage = 1
        elif seeds <= 5:
            force_stage = 2
        else:
            force_stage = 3
    return adaptive_eval(hid, force_stage=force_stage, objectives=objectives)


def read_trace(hid: str, seed: int, *, lines: int = 50) -> List[dict]:
    """Return up to ``lines`` records from a trace JSONL."""
    p = HARNESS_DIR / hid / "traces" / f"seed_{seed:02d}.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)
    out: List[dict] = []
    with p.open() as fh:
        for i, line in enumerate(fh):
            if i >= lines:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def trace_summary(hid: str) -> dict:
    p = HARNESS_DIR / hid / "traces" / "summary.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


def _harness_files(hid: str) -> dict[str, str]:
    hd = HARNESS_DIR / hid / "harness"
    out: dict[str, str] = {}
    if hd.exists():
        for f in sorted(hd.rglob("*.py")):
            out[str(f.relative_to(hd))] = f.read_text()
    return out


def diff_harnesses(hid_a: str, hid_b: str) -> str:
    fa = _harness_files(hid_a)
    fb = _harness_files(hid_b)
    keys = sorted(set(fa) | set(fb))
    parts: list[str] = []
    for k in keys:
        a = fa.get(k, "").splitlines(keepends=True)
        b = fb.get(k, "").splitlines(keepends=True)
        if a == b:
            continue
        parts.extend(difflib.unified_diff(
            a, b,
            fromfile=f"{hid_a}/{k}",
            tofile=f"{hid_b}/{k}",
            n=3,
        ))
    return "".join(parts)


# ---------------------------------------------------------------------------
# spawn (creates a new harness branch)
# ---------------------------------------------------------------------------


def spawn_branch(
    parent_hid: str,
    *,
    mode: Optional[str] = None,
    base_model: Optional[str] = None,
    rationale: str = "",
    edits_from: Optional[Path] = None,
    new_id: Optional[str] = None,
    n_agents: Optional[int] = None,
    map_kind: Optional[str] = None,
    env: Optional[str] = None,
) -> str:
    """Create a new harness branch under harnesses/<new_id>.

    The simplest path: ``edits_from`` is a directory with the proposer's
    new harness/ tree (already authored); the spawn helper copies it in,
    fills the manifest from the parent's defaults plus overrides, and
    writes the rationale.md.

    Without ``edits_from``, the parent's harness/ tree is copied
    unchanged — useful for a parameter-only branch (e.g., flipping
    base_model on an M2 scaffold).
    """
    parent_dir = HARNESS_DIR / parent_hid
    if not parent_dir.exists():
        raise FileNotFoundError(parent_hid)
    parent_manifest = json.loads((parent_dir / "manifest.json").read_text())

    new_id = new_id or next_id()
    new_dir = HARNESS_DIR / new_id
    if new_dir.exists():
        raise FileExistsError(new_dir)
    new_dir.mkdir(parents=True)

    # Copy harness source (unless edits_from provided).
    if edits_from is not None:
        edits_from = Path(edits_from).resolve()
        if not edits_from.exists():
            raise FileNotFoundError(edits_from)
        shutil.copytree(edits_from, new_dir / "harness", dirs_exist_ok=True)
    else:
        src_h = parent_dir / "harness"
        if src_h.exists():
            shutil.copytree(src_h, new_dir / "harness")
        else:
            (new_dir / "harness").mkdir()

    manifest = {
        "id": new_id,
        "mode": mode or parent_manifest.get("mode", "M3"),
        "base_model": base_model if base_model is not None
                      else parent_manifest.get("base_model"),
        "env": env or parent_manifest.get("env"),
        "n_agents": int(n_agents or parent_manifest.get("n_agents")),
        "map": map_kind or parent_manifest.get("map", "default"),
        "parent": parent_hid,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": "",
        "synthesis_token_cost": 0.0,
        "inference_token_cost": 0.0,
        "rationale_path": "rationale.md",
        "policy_path": parent_manifest.get("policy_path", "harness/policy.py"),
    }
    (new_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (new_dir / "rationale.md").write_text(
        f"# {new_id}\n\n**Parent**: {parent_hid}\n\n{rationale}\n"
    )
    return new_id


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


def write_probe(name: str, code: str) -> Path:
    if not re.match(r"^[A-Za-z0-9_\-]+$", name):
        raise ValueError("probe name must match [A-Za-z0-9_-]+")
    p = SCRATCH_DIR / f"{name}.py"
    _assert_writable(p, allow_existing=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(code)
    return p


def run_probe(name: str, args: List[str] | None = None) -> str:
    p = SCRATCH_DIR / f"{name}.py"
    if not p.exists():
        raise FileNotFoundError(p)
    args = args or []
    proc = subprocess.run(
        [sys.executable, str(p), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    return f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\nexit={proc.returncode}"


# ---------------------------------------------------------------------------
# log (proposer narrative)
# ---------------------------------------------------------------------------


def append_log(text: str) -> None:
    _assert_writable(LOG_PATH, allow_existing=True)
    ts = time.strftime("%Y-%m-%d %H:%M")
    with LOG_PATH.open("a") as fh:
        fh.write(f"\n## {ts}\n\n{text.rstrip()}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print(obj):
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2))
    else:
        print(obj)


def main():
    ap = argparse.ArgumentParser("autoresearch.meta.tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--env", default=None)
    p_list.add_argument("--mode", default=None)

    p = sub.add_parser("read"); p.add_argument("hid")
    p.add_argument("--no-source", action="store_true",
                   help="omit policy source from output")

    p = sub.add_parser("eval"); p.add_argument("hid")
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--objectives", default=None,
                   help="comma-separated objectives list")

    p = sub.add_parser("diff"); p.add_argument("hid_a"); p.add_argument("hid_b")

    p = sub.add_parser("pareto"); p.add_argument("--objectives", default=None)

    p = sub.add_parser("spawn")
    p.add_argument("parent")
    p.add_argument("--mode", default=None)
    p.add_argument("--base-model", default=None, dest="base_model")
    p.add_argument("--rationale", default="")
    p.add_argument("--edits-from", default=None, dest="edits_from")
    p.add_argument("--id", default=None, dest="new_id")
    p.add_argument("--env", default=None)
    p.add_argument("--n-agents", type=int, default=None, dest="n_agents")
    p.add_argument("--map", default=None, dest="map_kind")

    p = sub.add_parser("trace")
    p.add_argument("hid")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lines", type=int, default=50)

    p = sub.add_parser("summary"); p.add_argument("hid")

    p = sub.add_parser("probe")
    sp = p.add_subparsers(dest="probecmd", required=True)
    pw = sp.add_parser("write"); pw.add_argument("name")
    pw.add_argument("--code-from", default="-",
                    help="path or '-' for stdin")
    pr = sp.add_parser("run"); pr.add_argument("name")
    pr.add_argument("args", nargs="*")

    p = sub.add_parser("log"); p.add_argument("text", nargs="?", default="-")

    sub.add_parser("next-id")
    sub.add_parser("cull")
    p_val = sub.add_parser("validate")
    p_val.add_argument("hid")

    args = ap.parse_args()

    try:
        if args.cmd == "list":
            _print(list_harnesses(filter_env=args.env, filter_mode=args.mode))
        elif args.cmd == "read":
            data = read_harness(args.hid)
            if args.no_source:
                data.pop("source", None)
            _print(data)
        elif args.cmd == "eval":
            obj = args.objectives.split(",") if args.objectives else None
            _print(eval_harness(args.hid, seeds=args.seeds, objectives=obj))
        elif args.cmd == "diff":
            sys.stdout.write(diff_harnesses(args.hid_a, args.hid_b))
        elif args.cmd == "pareto":
            from autoresearch.meta.population import pareto_frontier
            obj = args.objectives.split(",") if args.objectives else None
            _print(pareto_frontier(objectives=obj))
        elif args.cmd == "spawn":
            edits = Path(args.edits_from) if args.edits_from else None
            new_id = spawn_branch(
                args.parent, mode=args.mode, base_model=args.base_model,
                rationale=args.rationale, edits_from=edits, new_id=args.new_id,
                env=args.env, n_agents=args.n_agents, map_kind=args.map_kind,
            )
            _print({"new_id": new_id})
        elif args.cmd == "trace":
            for rec in read_trace(args.hid, args.seed, lines=args.lines):
                print(json.dumps(rec))
        elif args.cmd == "summary":
            _print(trace_summary(args.hid))
        elif args.cmd == "probe":
            if args.probecmd == "write":
                code = (sys.stdin.read() if args.code_from == "-"
                        else Path(args.code_from).read_text())
                p = write_probe(args.name, code)
                _print(str(p))
            else:
                print(run_probe(args.name, args.args))
        elif args.cmd == "log":
            text = sys.stdin.read() if args.text == "-" else args.text
            append_log(text)
            _print("ok")
        elif args.cmd == "next-id":
            _print(next_id())
        elif args.cmd == "cull":
            from autoresearch.meta.population import cull_dominated
            _print({"archived": cull_dominated()})
        elif args.cmd == "validate":
            from pipeline.harness import validate_harness
            ok, msg = validate_harness(HARNESS_DIR / args.hid)
            _print({"ok": ok, "msg": msg})
        else:
            ap.error(f"unknown command {args.cmd}")
    except (FileNotFoundError, FileExistsError, PermissionError, ValueError) as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
