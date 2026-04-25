#!/usr/bin/env bash
#
# autoresearch/meta/archive_run.sh
#
# Archive a completed meta-harness run's artifacts into
# ``autoresearch/meta_archive/<tag>/`` and (optionally) reset the
# working tree to a baseline commit so the next run starts clean.
#
# What gets archived:
#   * autoresearch/meta/harnesses/             — all harnesses (incl. seed)
#   * autoresearch/meta/scratch/               — proposer probes
#   * autoresearch/meta/archive/               — culled harnesses
#   * autoresearch/meta/figures/               — proposer-generated plots
#   * autoresearch/meta/pareto.json            — frontier state
#   * autoresearch/meta/proposer_log.md        — narrative
#   * autoresearch/meta/queue.jsonl            — queue
#   * HARN_RUN*.md (repo root)                 — analysis notes
#
# What does NOT get archived (framework code, kept in place / restored
# by the reset):
#   * autoresearch/meta/{tools,evaluator,population,__init__}.py
#   * autoresearch/meta/{program.md,run_experiment.sh,archive_run.sh}
#   * pipeline/{trace,harness}.py
#   * envs, llm_self_play.py, run_inner_loop.py
#
# The archive directory is in ``.gitignore`` so it survives
# ``git reset --hard``. After archiving (and optional reset), the
# working tree is back to the baseline state and the next run can be
# launched with a clean slate.
#
# Usage:
#   archive_run.sh <tag> [--reset-to <commit>] [--dry-run] [--force]
#                  [--no-prompt]
#
# Examples:
#   # Archive the current state to autoresearch/meta_archive/run1/
#   ./autoresearch/meta/archive_run.sh run1
#
#   # Archive AND reset to the pre-run baseline commit
#   ./autoresearch/meta/archive_run.sh run1 \
#       --reset-to 9e554aae2a7cca40e99499d8a953bab2127b1f40
#
#   # See what would be archived without copying anything
#   ./autoresearch/meta/archive_run.sh run1 --dry-run
#

set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

TAG=""
RESET_TO=""
DRY_RUN=0
FORCE=0
NO_PROMPT=0

usage() {
    sed -n 's/^# \{0,1\}//p' "$0" | sed -n '2,/^$/p'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reset-to)   RESET_TO="${2:?missing value for --reset-to}"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --force)      FORCE=1; shift ;;
        --no-prompt)  NO_PROMPT=1; shift ;;
        -h|--help)    usage 0 ;;
        --*)          echo "ERROR: unknown flag $1"; usage 1 ;;
        *)            if [[ -z "$TAG" ]]; then TAG="$1"; else echo "ERROR: extra arg $1"; usage 1; fi
                      shift ;;
    esac
done

if [[ -z "$TAG" ]]; then
    echo "ERROR: missing <tag>"
    usage 1
fi

if [[ ! "$TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "ERROR: tag must match [A-Za-z0-9_.-]+"
    exit 1
fi

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$REPO_ROOT" ]]; then
    echo "ERROR: not in a git repo"
    exit 1
fi
cd "$REPO_ROOT"

ARCHIVE_ROOT="autoresearch/meta_archive"
ARCHIVE_DIR="$ARCHIVE_ROOT/$TAG"

# Validate reset target before doing anything destructive.
if [[ -n "$RESET_TO" ]]; then
    if ! git rev-parse --verify --quiet "$RESET_TO" >/dev/null; then
        echo "ERROR: --reset-to ref '$RESET_TO' is not a valid commit"
        exit 1
    fi
    RESET_SHA="$(git rev-parse "$RESET_TO")"
fi

if [[ -e "$ARCHIVE_DIR" ]]; then
    if [[ "$FORCE" -eq 1 ]]; then
        if [[ "$DRY_RUN" -eq 0 ]]; then
            rm -rf "$ARCHIVE_DIR"
        fi
    else
        echo "ERROR: $ARCHIVE_DIR already exists. Use --force to overwrite."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Build artifact file list
# ---------------------------------------------------------------------------
#
# We enumerate by walking the artifact directories rather than relying on
# git status, so the script works regardless of whether the proposer
# committed during the run.

TMP_LIST="$(mktemp)"
trap 'rm -f "$TMP_LIST"' EXIT

# Directories under autoresearch/meta/ to capture in full.
META_DIRS=(
    "autoresearch/meta/harnesses"
    "autoresearch/meta/scratch"
    "autoresearch/meta/archive"
    "autoresearch/meta/figures"
)
# Single files under autoresearch/meta/ to capture if present.
META_FILES=(
    "autoresearch/meta/pareto.json"
    "autoresearch/meta/proposer_log.md"
    "autoresearch/meta/queue.jsonl"
)
# Framework files under autoresearch/meta/ that are NOT artifacts —
# the proposer is forbidden from modifying these. Any other top-level
# .py / .md / .sh in autoresearch/meta/ is treated as a proposer-authored
# run artifact and archived.
META_FRAMEWORK_FILES=(
    "autoresearch/meta/__init__.py"
    "autoresearch/meta/tools.py"
    "autoresearch/meta/evaluator.py"
    "autoresearch/meta/population.py"
    "autoresearch/meta/program.md"
    "autoresearch/meta/run_experiment.sh"
    "autoresearch/meta/archive_run.sh"
)
# Repo-root glob patterns.
ROOT_GLOBS=(
    "HARN_RUN*.md"
)

for d in "${META_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
        # exclude the archive root in case it was somehow inside one of these.
        find "$d" -type f -not -path "$ARCHIVE_ROOT/*" >> "$TMP_LIST"
    fi
done

for f in "${META_FILES[@]}"; do
    [[ -f "$f" ]] && echo "$f" >> "$TMP_LIST"
done

# Catch any *unknown* top-level file in autoresearch/meta/ — these are
# proposer-authored helpers (e.g. analysis scripts, ad-hoc notes) that
# don't belong to the framework whitelist.
shopt -s nullglob
for f in autoresearch/meta/*.py autoresearch/meta/*.md autoresearch/meta/*.sh \
         autoresearch/meta/*.json autoresearch/meta/*.jsonl \
         autoresearch/meta/*.txt; do
    [[ -f "$f" ]] || continue
    is_framework=0
    for fw in "${META_FRAMEWORK_FILES[@]}"; do
        if [[ "$f" == "$fw" ]]; then is_framework=1; break; fi
    done
    [[ "$is_framework" -eq 1 ]] && continue
    # Skip files already enumerated via META_FILES.
    already_listed=0
    for known in "${META_FILES[@]}"; do
        if [[ "$f" == "$known" ]]; then already_listed=1; break; fi
    done
    [[ "$already_listed" -eq 1 ]] && continue
    echo "$f" >> "$TMP_LIST"
done
shopt -u nullglob

shopt -s nullglob
for pat in "${ROOT_GLOBS[@]}"; do
    for f in $pat; do
        [[ -f "$f" ]] && echo "$f" >> "$TMP_LIST"
    done
done
shopt -u nullglob

# Defensively strip anything inside the archive root.
grep -v "^${ARCHIVE_ROOT}/" "$TMP_LIST" > "${TMP_LIST}.f" || true
mv "${TMP_LIST}.f" "$TMP_LIST"

N_FILES=$(wc -l < "$TMP_LIST" | tr -d ' ')

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

HEAD_SHA="$(git rev-parse HEAD)"
BRANCH="$(git branch --show-current 2>/dev/null || echo '<detached>')"

cat <<EOF
[archive_run]
  tag:          $TAG
  repo:         $REPO_ROOT
  branch:       $BRANCH
  head:         $HEAD_SHA
  reset_to:     ${RESET_TO:-<none>}
  archive_dir:  $ARCHIVE_DIR
  files:        $N_FILES
  dry_run:      $DRY_RUN
EOF

if [[ "$N_FILES" -eq 0 ]]; then
    echo "WARNING: no artifacts found under the configured paths."
    echo "         Either the run produced no files or paths are misconfigured."
    if [[ -z "$RESET_TO" ]]; then
        echo "Nothing to do."
        exit 0
    fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ""
    echo "--- files that WOULD be archived ---"
    head -30 "$TMP_LIST"
    if [[ "$N_FILES" -gt 30 ]]; then
        echo "... ($((N_FILES - 30)) more)"
    fi
    if [[ -n "$RESET_TO" ]]; then
        echo ""
        echo "--- WOULD reset working tree to $RESET_SHA ---"
        echo "Tracked files would be reverted to that commit's state."
        echo "Untracked artifact files (above) would be removed after archiving."
    fi
    exit 0
fi

# Confirm destructive operation if reset requested.
if [[ -n "$RESET_TO" && "$NO_PROMPT" -eq 0 ]]; then
    echo ""
    echo "About to archive $N_FILES files AND reset to $RESET_SHA."
    echo -n "Proceed? [y/N] "
    read -r answer
    case "$answer" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

# ---------------------------------------------------------------------------
# Copy artifacts into the archive
# ---------------------------------------------------------------------------

mkdir -p "$ARCHIVE_DIR"

while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    target="$ARCHIVE_DIR/$f"
    mkdir -p "$(dirname "$target")"
    cp -p "$f" "$target"
done < "$TMP_LIST"

# Metadata sidecar.
{
    echo "tag: $TAG"
    echo "timestamp_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo: $REPO_ROOT"
    echo "branch: $BRANCH"
    echo "head_sha: $HEAD_SHA"
    echo "reset_to: ${RESET_TO:-<none>}"
    echo "reset_sha: ${RESET_SHA:-<none>}"
    echo "n_files: $N_FILES"
} > "$ARCHIVE_DIR/_archive_metadata.txt"

# If HEAD is ahead of the reset target, save the commit log between them.
if [[ -n "$RESET_TO" ]]; then
    if ! git merge-base --is-ancestor "$RESET_TO" HEAD 2>/dev/null; then
        echo "WARNING: HEAD is not a descendant of $RESET_TO."
        echo "         git reset --hard will move HEAD backwards across a fork."
    fi
    git log --oneline "$RESET_TO..HEAD" > "$ARCHIVE_DIR/_run_commits.txt" 2>/dev/null \
        || true
fi

echo "[archive_run] archived $N_FILES files to $ARCHIVE_DIR"

# ---------------------------------------------------------------------------
# Optional reset
# ---------------------------------------------------------------------------

if [[ -n "$RESET_TO" ]]; then
    # Record the previous HEAD so the user can recover via reflog if needed.
    PREV_HEAD="$HEAD_SHA"

    git reset --hard "$RESET_TO"

    # ``git reset --hard`` reverts tracked files but does NOT remove the
    # files we copied that were untracked. Walk the artifact list and
    # rm anything that still exists and is untracked.
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        if [[ -e "$f" ]]; then
            if ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
                rm -f "$f"
            fi
        fi
    done < "$TMP_LIST"

    # Remove now-empty proposer-created directories under autoresearch/meta/
    # but leave the *tracked* directory skeleton intact (scratch/, archive/,
    # harnesses/h0000/ etc.).
    find autoresearch/meta -type d -empty -not -path "$ARCHIVE_ROOT*" \
        -delete 2>/dev/null || true

    NEW_HEAD="$(git rev-parse HEAD)"
    cat <<EOF
[archive_run] reset done.
  HEAD was:  $PREV_HEAD
  HEAD now:  $NEW_HEAD
  Archive:   $ARCHIVE_DIR

To recover the pre-reset state if needed:
  git reflog | head
  git reset --hard $PREV_HEAD
EOF
else
    echo "[archive_run] skipped reset (no --reset-to given)."
fi
