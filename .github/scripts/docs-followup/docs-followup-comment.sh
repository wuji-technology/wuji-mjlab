#!/usr/bin/env bash
# docs-followup-comment.sh
#
# Post a docs follow-up prompt comment on the merged code PR and tag the PR
# with the `docs:followup` label. Run by the post-merge-docs-followup workflow
# when the merged PR touched the repository's public surface.
#
# The comment body is the ready-to-use prompt the docs maintainer feeds to
# Claude; the label marks the PR as outstanding docs debt until the callback
# workflow clears it.
#
# Required environment variables:
#   GH_TOKEN     Token with `pull-requests: write` on REPO
#   REPO         owner/name of the repository
#   PR_NUMBER    The merged code PR number
#   PR_TITLE     The merged code PR title
#   PR_AUTHOR    GitHub login of the merged code PR author

set -euo pipefail

: "${GH_TOKEN:?GH_TOKEN env required}"
: "${REPO:?REPO env required}"
: "${PR_NUMBER:?PR_NUMBER env required}"
: "${PR_TITLE:?PR_TITLE env required}"
: "${PR_AUTHOR:?PR_AUTHOR env required}"

FOLLOWUP_LABEL="docs:followup"

# Ensure the label exists. `gh pr edit --add-label` fails outright if the label
# is missing from the repo, so create it lazily on first run (idempotent — a
# pre-existing label makes `gh label create` non-zero and we swallow that).
gh label create "$FOLLOWUP_LABEL" -R "$REPO" \
  --color "fbca04" \
  --description "对外公开面变更，待跟进文档" \
  >/dev/null 2>&1 || true

# Tag the PR first so it surfaces in the docs-debt list even if the comment
# step fails and the workflow falls back. Use the REST API directly instead of
# `gh pr edit --add-label`: gh CLI 2.x's pr edit incidentally queries Projects
# (classic), which GitHub is sunsetting and now returns a GraphQL deprecation
# error → exit 1, even though the label itself would have been added fine.
gh api -X POST "repos/$REPO/issues/$PR_NUMBER/labels" \
  -f "labels[]=$FOLLOWUP_LABEL" \
  >/dev/null

BODY_FILE="$(mktemp)"
trap 'rm -f "$BODY_FILE"' EXIT

# The full Claude prompt now ships in the Feishu DM card (followup_needed)
# rendered as a copyable code block; keeping it in two places risked drift.
# This PR comment stays as the public anchor: the docs:followup label, the
# "what to do" steps, and the /docs-done|/docs-skip handoff back to the
# callback workflow.
#
# Variable references below expand to their literal values only; bash does not
# re-scan substituted values, so a PR title containing backticks is inert.
cat > "$BODY_FILE" <<EOF
## 📝 文档跟进 · 本 PR 影响对外公开面

本 PR 已合入 \`main\`，改动触及对外公开面（公开 API、配置、协议、对外文档等），需要判断对外文档是否要更新。

### 怎么做
1. 完整 Claude prompt 已发到飞书 DM，从卡片代码块整段复制后在本仓库本地喂给 Claude（Claude Code）
2. 需要改 → 直接改对外文档并提文档 PR，reviewer 设为 @${PR_AUTHOR}
3. 收尾 → 在本 PR 评论 \`/docs-done\`（已提文档 PR）或 \`/docs-skip\`（判定无需改动），自动清除 \`docs:followup\` 标签

---

> 由 post-merge-docs-followup 自动发布。带 \`docs:followup\` 标签的 PR = 待跟进的文档债。
EOF

gh pr comment "$PR_NUMBER" -R "$REPO" --body-file "$BODY_FILE"

echo "Posted docs follow-up comment + label on $REPO #$PR_NUMBER"
