#!/usr/bin/env bash
# feishu-dm.sh
#
# Send a docs-followup notification card to the Feishu DM group via a signed
# custom-bot webhook. Reads webhook URL and signing secret from the environment.
#
# Required environment variables:
#   FEISHU_DOCS_DM_WEBHOOK       Full webhook URL of the custom bot
#   FEISHU_DOCS_DM_SIGN_SECRET   HMAC signing secret of the custom bot
#
# Required flags:
#   --pr <number>              The code PR number that triggered this run
#   --status <kind>            One of:
#                                internal_change | followup_needed | failed
#                                done | skipped     (callback outcomes)
#
# Optional flags (used to enrich the card depending on status):
#   --repo <owner/name>        Repository slug; defaults to $GITHUB_REPOSITORY.
#                              Used for the PR link and the card title.
#   --pr-title <title>         Title of the original code PR
#   --pr-author <login>        GitHub login of the original code PR author
#   --run-url <url>            Workflow run URL (status=failed)
#   --docs-pr-url <url>        跟进文档 PR 链接，渲染为独立的「文档 PR」行。
#                              status=done 时必填——done 卡是维护者收到的跟进
#                              完成记录，缺链接直接拒发。
#   --detail <text>            Maintainer-supplied outcome note appended to the
#                              card body. For callback statuses (done/skipped)
#                              this is the text after the command in the
#                              trigger comment — e.g. a skip rationale or
#                              grep evidence. Empty string hides the row.
#   --weekly-json <path>       Bucketed weekly debt JSON produced by
#                              docs-followup-weekly.sh. Required when
#                              status=weekly_report; ignored otherwise.
#
# Exits non-zero if Feishu rejects the message (StatusCode != 0).

set -euo pipefail

PR_NUMBER=""
STATUS=""
REPO="${GITHUB_REPOSITORY:-}"
PR_TITLE=""
PR_AUTHOR=""
RUN_URL=""
DOCS_PR_URL=""
DETAIL=""
WEEKLY_JSON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pr) PR_NUMBER="$2"; shift 2 ;;
    --status) STATUS="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --pr-title) PR_TITLE="$2"; shift 2 ;;
    --pr-author) PR_AUTHOR="$2"; shift 2 ;;
    --run-url) RUN_URL="$2"; shift 2 ;;
    --docs-pr-url) DOCS_PR_URL="$2"; shift 2 ;;
    --detail) DETAIL="$2"; shift 2 ;;
    --weekly-json) WEEKLY_JSON="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

: "${STATUS:?--status required}"
# weekly_report is an org-wide digest; it doesn't tie to a single PR, so PR/REPO
# requirements only kick in for per-PR statuses. Caller still passes `--pr 0`
# by convention so older tooling that greps for `--pr` doesn't trip.
if [[ "$STATUS" != "weekly_report" ]]; then
  : "${PR_NUMBER:?--pr required (use --pr 0 for weekly_report)}"
  : "${REPO:?--repo required (or set GITHUB_REPOSITORY)}"
fi

# Soft-fail when bot credentials are missing: surface a workflow warning so the
# maintainer notices, but do not exit non-zero. A hard failure here gets masked
# by `continue-on-error: true` on the caller, making the misconfiguration
# silently invisible — the warning channel is louder than a swallowed error.
if [[ -z "${FEISHU_DOCS_DM_WEBHOOK:-}" || -z "${FEISHU_DOCS_DM_SIGN_SECRET:-}" ]]; then
  echo "::warning title=Feishu DM skipped::FEISHU_DOCS_DM_WEBHOOK / FEISHU_DOCS_DM_SIGN_SECRET not set on $REPO — configure them in Settings → Secrets and variables → Actions to enable docs-followup notifications."
  echo "Feishu DM skipped: bot credentials not configured" >&2
  exit 0
fi

REPO_NAME="${REPO##*/}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%s)"
SIGN="$("$SCRIPT_DIR/feishu-sign.sh" "$TIMESTAMP" "$FEISHU_DOCS_DM_SIGN_SECRET")"

case "$STATUS" in
  internal_change)
    HEADER_TITLE="📭 Docs Follow-up · ${REPO_NAME} #${PR_NUMBER}"
    HEADER_TEMPLATE="grey"
    BODY_LINE="判定本次改动不需要文档跟进。若觉得判断有误，可手动追加文档更新。"
    ;;
  followup_needed)
    HEADER_TITLE="📝 Docs Follow-up · ${REPO_NAME} #${PR_NUMBER}"
    HEADER_TEMPLATE="blue"
    BODY_LINE="影响对外公开面，需要文档跟进。请在本地用 Claude 分析并提交文档跟进 PR。"
    ;;
  done)
    HEADER_TITLE="✅ Docs Follow-up · ${REPO_NAME} #${PR_NUMBER}"
    HEADER_TEMPLATE="green"
    BODY_LINE="文档跟进已完成，请查看文档 PR。"
    # done 卡必须带文档 PR 链接。
    if [[ -z "$DOCS_PR_URL" ]]; then
      echo "--docs-pr-url required when --status done" >&2
      exit 2
    fi
    ;;
  skipped)
    HEADER_TITLE="📭 Docs Follow-up · ${REPO_NAME} #${PR_NUMBER}"
    HEADER_TEMPLATE="grey"
    BODY_LINE="判定无需改动文档，跟进流程收尾。"
    ;;
  failed)
    HEADER_TITLE="❌ Docs Follow-up · ${REPO_NAME} #${PR_NUMBER}"
    HEADER_TEMPLATE="red"
    BODY_LINE="自动检测流程失败，请手动确认是否需要更新文档。"
    ;;
  weekly_report)
    HEADER_TITLE="📊 Docs Follow-up 周报"
    HEADER_TEMPLATE="purple"
    BODY_LINE=""  # weekly card builds its own body from --weekly-json
    : "${WEEKLY_JSON:?--weekly-json <path> required when --status weekly_report}"
    [[ -s "$WEEKLY_JSON" ]] || { echo "weekly json $WEEKLY_JSON is empty" >&2; exit 2; }
    ;;
  *)
    echo "Unknown status: $STATUS" >&2; exit 2 ;;
esac

# Build the Feishu interactive-card JSON payload: timestamp, HMAC signature,
# the status header, and the status-dependent body elements.
build_card() {
  local elements_json
  elements_json="$(
    jq -n \
      --arg body "$BODY_LINE" \
      --arg repo "$REPO" \
      --arg pr_title "$PR_TITLE" \
      --arg pr_author "$PR_AUTHOR" \
      --arg run_url "$RUN_URL" \
      --arg pr_number "$PR_NUMBER" \
      --arg docs_pr_url "$DOCS_PR_URL" \
      --arg detail "$DETAIL" \
      '
      def kv(name; value):
        if value == "" then empty
        else { "tag": "div", "text": { "tag": "lark_md", "content": "**\(name)：**\(value)" } }
        end;

      [
        { "tag": "div", "text": { "tag": "lark_md", "content": $body } },
        kv("仓库"; $repo),
        kv("原 PR"; if $pr_title == "" then "" else "[\($pr_title)](https://github.com/\($repo)/pull/\($pr_number))" end),
        kv("作者"; if $pr_author == "" then "" else "@\($pr_author)" end),
        kv("文档 PR"; if $docs_pr_url == "" then "" else "[\($docs_pr_url)](\($docs_pr_url))" end),
        kv("结论"; $detail),
        kv("Workflow"; if $run_url == "" then "" else "[运行详情](\($run_url))" end)
      ]
      '
  )"

  jq -n \
    --arg ts "$TIMESTAMP" \
    --arg sign "$SIGN" \
    --arg title "$HEADER_TITLE" \
    --arg template "$HEADER_TEMPLATE" \
    --argjson elements "$elements_json" \
    '{
      timestamp: $ts,
      sign: $sign,
      msg_type: "interactive",
      card: {
        config: { wide_screen_mode: true },
        header: {
          title: { tag: "plain_text", content: $title },
          template: $template
        },
        elements: $elements
      }
    }'
}

# Weekly digest renders three buckets from the snapshot JSON instead of the
# per-PR header/body/fields format. Keep it as a separate builder so the
# per-PR card path stays unchanged and easy to reason about.
build_weekly_card() {
  local elements_json
  elements_json="$(
    jq -n \
      --slurpfile data "$WEEKLY_JSON" \
      '
      ($data[0]) as $d |

      # Render a PR list bucket. Empty bucket -> a "（无）" placeholder so the
      # card layout stays predictable across weeks.
      def pr_lines(items; show_age):
        if (items | length) == 0 then "（无）"
        else
          items
          | map(
              "- [\(.repository.nameWithOwner)#\(.number)](\(.url)) \(.title)"
              + (if show_age then "  · 挂 \(.age_days) 天" else "" end)
            )
          | join("\n")
        end;

      [
        { "tag": "div", "text": { "tag": "lark_md",
            "content": "**总览：**当前挂账 **\($d.total)** 条 · 本周新增 \($d.new_this_week | length) · 本周收尾 \($d.closed_in_week | length) · Stale (>14天) \($d.stale_debt | length)" } },
        { "tag": "hr" },
        { "tag": "div", "text": { "tag": "lark_md",
            "content": ("**🆕 本周新增挂账**\n" + pr_lines($d.new_this_week; false)) } },
        { "tag": "hr" },
        { "tag": "div", "text": { "tag": "lark_md",
            "content": ("**✅ 本周收尾**\n" + pr_lines($d.closed_in_week; false)) } },
        { "tag": "hr" },
        { "tag": "div", "text": { "tag": "lark_md",
            "content": ("**⏳ Stale debt**（按挂账天数倒序）\n" + pr_lines($d.stale_debt; true)) } }
      ]
      '
  )"

  jq -n \
    --arg ts "$TIMESTAMP" \
    --arg sign "$SIGN" \
    --arg title "$HEADER_TITLE" \
    --arg template "$HEADER_TEMPLATE" \
    --argjson elements "$elements_json" \
    '{
      timestamp: $ts,
      sign: $sign,
      msg_type: "interactive",
      card: {
        config: { wide_screen_mode: true },
        header: {
          title: { tag: "plain_text", content: $title },
          template: $template
        },
        elements: $elements
      }
    }'
}

if [[ "$STATUS" == "weekly_report" ]]; then
  PAYLOAD="$(build_weekly_card)"
else
  PAYLOAD="$(build_card)"
fi

RESPONSE="$(
  curl -fsS --max-time 30 -X POST \
    -H "Content-Type: application/json" \
    --data "$PAYLOAD" \
    "$FEISHU_DOCS_DM_WEBHOOK"
)"

CODE="$(echo "$RESPONSE" | jq -r '.StatusCode // .code // "missing"')"
if [[ "$CODE" != "0" ]]; then
  echo "Feishu rejected webhook (code=$CODE): $RESPONSE" >&2
  exit 1
fi

echo "Feishu DM sent: $STATUS"
