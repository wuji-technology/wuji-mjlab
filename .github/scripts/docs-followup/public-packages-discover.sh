#!/usr/bin/env bash
# public-packages-discover.sh
#
# Emit one path prefix per line for paths considered "public" — i.e., changes
# under any of these paths should trigger docs follow-up analysis. The workflow
# turns each line into an anchored prefix (`^<prefix>`) before matching, so
# entries that are substrings of each other do not cross-match.
#
# Source:
#   - Static list in <caller-repo>/.github/docs-sync-public-packages.txt
#     (authoritative, per-repo).
#
# Why no dynamic discovery (e.g. scanning package.json / Cargo.toml flags):
# manifest flags rarely map cleanly to "user-facing". A static list per repo
# is the single source of truth; this script abstracts the lookup so the
# reusable workflow can call it without caring how each repo enumerates its
# public surface.
#
# Working directory contract:
#   In the reusable workflow this script lives in
#   `.docs-center/scripts/docs-followup/`, but the public-package list belongs
#   to the *caller* repository at `.github/docs-sync-public-packages.txt`.
#   Default to `$PWD` (the caller repo root, which the reusable workflow
#   guarantees as cwd) and let CALLER_REPO_ROOT override it for explicit
#   callers / local smoke tests.

set -euo pipefail

REPO_ROOT="${CALLER_REPO_ROOT:-$PWD}"
STATIC_LIST="$REPO_ROOT/.github/docs-sync-public-packages.txt"

# Emit each non-comment, non-blank path prefix from the static list file.
emit_static() {
  [[ -f "$STATIC_LIST" ]] || return 0
  awk '!/^[[:space:]]*(#|$)/' "$STATIC_LIST"
}

emit_static | awk 'NF && !seen[$0]++'
