#!/usr/bin/env bash
# force-followup-discover.sh
#
# Emit one path prefix per line for paths that should ALWAYS trigger docs
# follow-up, even when the PR title prefix would normally classify the change
# as internal-only (chore/test/refactor/perf/docs/style/ci, including the
# compound `ci/docs:` form).
#
# Companion to `public-packages-discover.sh`. The reusable workflow consults
# this list AFTER confirming the change touches a public path, and BEFORE the
# title-prefix skip — any hit here bypasses the title-prefix rule.
#
# Source:
#   - Static list in <caller-repo>/.github/docs-sync-force-followup.txt
#     (optional; per-repo; missing file ⇒ no force paths)
#
# Why this list is optional:
# Most repos can rely on the title-prefix heuristic. The force list exists for
# high-signal surfaces where prior PRs have shown the heuristic misfires
# (e.g. Python .pyi stubs where `docs:` changes carry real API impact).
# Repos without such surfaces should omit the file entirely.
#
# Working directory contract: same as public-packages-discover.sh — defaults
# to $PWD (the caller repo root, which the reusable workflow guarantees as
# cwd) and lets CALLER_REPO_ROOT override for explicit callers / local tests.

set -euo pipefail

REPO_ROOT="${CALLER_REPO_ROOT:-$PWD}"
STATIC_LIST="$REPO_ROOT/.github/docs-sync-force-followup.txt"

emit_static() {
  [[ -f "$STATIC_LIST" ]] || return 0
  awk '!/^[[:space:]]*(#|$)/' "$STATIC_LIST"
}

emit_static | awk 'NF && !seen[$0]++'
