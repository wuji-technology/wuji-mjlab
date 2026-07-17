#!/usr/bin/env bash
# docs-followup-prompt.sh
#
# Render the docs follow-up Claude prompt for a merged code PR.
# Self-hosted variant for wuji-mjlab — hardcoded product config (in-hand
# reorientation RL, pixi task surface). No dependency on wuji-docs-center
# repos.json.
#
# Required environment variables:
#   REPO         owner/name of the repository
#   PR_NUMBER    The merged code PR number
#   PR_TITLE     The merged code PR title
#   PR_AUTHOR    GitHub login of the merged code PR author

set -euo pipefail

: "${REPO:?REPO env required (owner/name)}"
: "${PR_NUMBER:?PR_NUMBER env required}"
: "${PR_TITLE:?PR_TITLE env required}"
: "${PR_AUTHOR:?PR_AUTHOR env required}"

PRODUCT_NAME="Wuji MJLab"
DOCS_PATH="docs/external"
SURFACE_HINT="pixi 任务入口（\`pixi run train\` / \`play\` 等）、任务名与 \`--task\` 取值、命令行参数、任务与环境配置项、导出与 sim2real 部署脚本、安装步骤与依赖声明"
NOISE_HINT="奖励函数与超参内部调整、算法实现重构、日志、测试、CI 等不改变用户可调接口的改动"
EXTRA_PATHS="\`CHANGELOG.md\`、\`README.md\`、\`pixi.toml\`、\`pyproject.toml\`、\`scripts/\`、\`deploy/\`"
CHANGELOG_HINT="条目按用户可执行的 pixi 任务与参数描述，不引用内部算法实现"

# Defensive: GitHub PR titles can in principle contain triple backticks, which
# would prematurely close the Feishu fenced code block this prompt renders into.
PR_TITLE_SAFE="${PR_TITLE//\`\`\`/ʼʼʼ}"

cat <<EOF
你是 ${PRODUCT_NAME} 文档维护助手。仓库 ${REPO} 的 PR #${PR_NUMBER}（${PR_TITLE_SAFE}，作者 @${PR_AUTHOR}）已合入 main。

任务：
1. 取代码改动：运行 \`gh pr diff ${PR_NUMBER} -R ${REPO}\`，并读 PR 描述
   同时搜索并阅读受影响代码的测试——行为常被 PR 未改动的既有测试钉死，这些测试不在 diff 里。找不到相关测试时，在结论中写明"未找到测试，仅依据代码核实"
2. 读取本仓库对外文档目录的现有内容（\`${DOCS_PATH}/\`、${EXTRA_PATHS}）
3. 判断这次改动是否产生用户可感知变化：${SURFACE_HINT}
4. 需要更新 → 修改对应文档文件；文档缺失而变更需要新文档 → 新建
5. 不产生用户可感知变化 → 回复"无需改动"并说明理由

判断规则：
- 只关注用户可感知变更（${SURFACE_HINT}）
- 忽略 ${NOISE_HINT}
- 写入对外文档的行为性表述必须从代码与测试重新推导，不得成句复用 CHANGELOG 原文。changelog 条目常把多条执行路径压进一句话，分号从句与"仍 / 但 / 除外"尾缀是高危信号，必须拆开逐条核实主语（哪条执行路径、哪个任务）与谓语（拦截 / 回落 / 报错）是否都来自代码事实

对外性判定（判断"用户可感知"的依据，2026-07-09 强化）：
- 依据是本仓对外面（注册任务 ID、pixi 任务入口、命令行参数、部署脚本与配置）里的公开入口 + PR 的 Breaking / 新任务标记，不是 CHANGELOG 是否已记。changelog 缺失是"应记未记"，不能反推"无需跟进"
- Breaking 变更、任务新增 / 默认行为变更：即使 changelog 未记、即使当前文档无对应章节，也要跟进（无章节就新建承载），不可判 skip
- 确属对外但 CHANGELOG [Unreleased] 未记 → 在跟进结论里提醒源 PR 作者补记，这是发布记录的唯一取材源，漏记会导致发版漏条目，但缺失本身不作为 skip 理由

版本发布记录（与正文承载独立判定，2026-07-09 起）：
- 本仓 docs/external 不承载 release-notes 页面，版本发布记录由发版流程从本仓 CHANGELOG.md 聚合生成到文档中心对应产品页
- 跟进时确认源 PR 的用户可感知变化已记入 CHANGELOG.md 的 [Unreleased] 段（以 origin/main 为准）：缺失 → 在跟进报告里点名提醒源 PR 作者补记，发布记录以 [Unreleased] 为唯一取材来源，漏记会导致发版漏条目
- 已随既往版本发布的变化无需处理

分支与合入目标（发布列车）：
- 判定需要改文档时，从 \`docs-prerelease\` 拉跟进分支，PR base 也用 \`docs-prerelease\`。线上文档由每日 tag 从各仓 main 构建，直接合 main 会把未发布功能的文档提前带上线。列车分支在双周发布日统一合入 main，main 的日常变更由保鲜 workflow 自动同步进列车分支
- 本仓是公开仓，受组织 ruleset 限制无法向上游推新分支：推到个人 fork，从 fork 向上游提 PR，base 仍是上游 \`docs-prerelease\`
- 涉及 docs-center 侧配套变更（meta.json 导航注册、reading-guide.json）时，同样走 wuji-docs-center 的 \`docs-prerelease\`

写作规范：
- 遵守 Microsoft Writing Style Guide
- 中文：删除"的/了/进行"等冗余词；用阿拉伯数字；主动语态、现在时
- 同一概念始终用同一术语；段落 3-7 行
- 不引入 frontmatter description 字段
- ${CHANGELOG_HINT}
EOF
