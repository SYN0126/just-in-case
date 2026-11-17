#!/bin/bash
# 双击上传本目录；使用 macOS 弹窗、GitHub CLI 和当前系统代理。
set -Eeuo pipefail
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"
export GH_HOST=github.com GH_PROMPT_DISABLED=1 GIT_TERMINAL_PROMPT=0
cd "$(dirname "${BASH_SOURCE[0]}")"
PROJECT_DIR=$(pwd -P)
DEFAULT_REPO="just-in-case"
ERROR_DETAIL="请查看终端中的错误信息，修复后再次双击重试。"
LOCK_OWNED=0

popup() {
    osascript - "$1" "$2" <<'APPLESCRIPT'
on run argv
    display dialog (item 2 of argv) buttons {"好"} default button "好" with title (item 1 of argv)
end run
APPLESCRIPT
}

ask() {
    osascript - "$1" "$2" "$3" "$4" <<'APPLESCRIPT'
on run argv
    set resultDialog to display dialog (item 2 of argv) default answer (item 3 of argv) buttons {"取消", item 4 of argv} default button (item 4 of argv) cancel button "取消" with title (item 1 of argv)
    return text returned of resultDialog
end run
APPLESCRIPT
}

finish() {
    local code=$?
    trap - EXIT
    if [ "$LOCK_OWNED" = 1 ]; then
        rm -f .github-upload.lock/pid
        rmdir .github-upload.lock 2>/dev/null || true
    fi
    if [ "$code" -ne 0 ]; then
        printf '\n✗ 上传未完成：%s\n' "$ERROR_DETAIL" >&2
        popup "GitHub 上传未完成" "$ERROR_DETAIL

已生成的提交仍保存在本地。详细原因请看终端。" >/dev/null 2>&1 || true
    fi
    exit "$code"
}
trap finish EXIT
trap 'exit 130' INT TERM

fail() { ERROR_DETAIL=$1; exit 1; }
step() { printf '\n[%s/4] %s\n' "$1" "$2"; }

# 仅在本进程内设置代理，不修改系统模式或全局 Git 配置。
configure_proxy() {
    local settings kind enabled host port
    PROXY_URL=${HTTPS_PROXY:-${https_proxy:-${ALL_PROXY:-${all_proxy:-${HTTP_PROXY:-${http_proxy:-}}}}}}
    if [ -z "$PROXY_URL" ]; then
        settings=$(scutil --proxy 2>/dev/null || true)
        for kind in HTTPS HTTP SOCKS; do
            enabled=$(printf '%s\n' "$settings" | awk -v k="${kind}Enable" '$1 == k {print $3; exit}')
            [ "$enabled" = 1 ] || continue
            host=$(printf '%s\n' "$settings" | awk -v k="${kind}Proxy" '$1 == k {print $3; exit}')
            port=$(printf '%s\n' "$settings" | awk -v k="${kind}Port" '$1 == k {print $3; exit}')
            [ -n "$host" ] && [[ "$port" =~ ^[0-9]+$ ]] || continue
            case "$host" in *:*) host="[$host]" ;; esac
            if [ "$kind" = SOCKS ]; then
                PROXY_URL="socks5h://$host:$port"
            else
                PROXY_URL="http://$host:$port"
            fi
            break
        done
    fi
    GIT_NETWORK=(-c credential.helper= -c 'credential.helper=!gh auth git-credential')
    if [ -n "$PROXY_URL" ]; then
        export HTTPS_PROXY="$PROXY_URL" HTTP_PROXY="$PROXY_URL" ALL_PROXY="$PROXY_URL"
        export https_proxy="$PROXY_URL" http_proxy="$PROXY_URL" all_proxy="$PROXY_URL"
        export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
        GIT_NETWORK+=(-c "http.proxy=$PROXY_URL" -c "remote.origin.proxy=$PROXY_URL")
        printf '  网络：已接入当前代理（支持规则 / 条件模式）\n'
    else
        printf '  网络：直连（未检测到已开启的系统代理）\n'
    fi
}

printf '\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n  CUDA 算子 · GitHub 同步\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
printf '  目录：%s\n  可见性：公开\n' "$PROJECT_DIR"
for tool in git gh osascript; do
    command -v "$tool" >/dev/null 2>&1 || fail "缺少 ${tool}。请先安装 Git 和 GitHub CLI（brew install gh）。"
done
configure_proxy

step 1 "检查 GitHub 登录与连接"
ERROR_DETAIL="无法连接 GitHub。请确认代理已启动，规则允许 GitHub；未登录时请在终端运行 gh auth login --hostname github.com --git-protocol https --web。"
LOGIN=$(gh api user --jq .login)
printf '  账号：%s\n' "$LOGIN"

# 诊断模式只读：可检查 API 和 Git 通路，不创建仓库或提交。
if [ "${1:-}" = --check ]; then
    git "${GIT_NETWORK[@]}" ls-remote https://github.com/cli/cli.git HEAD
    printf '\n✓ GitHub API 与 Git 连接正常。\n'
    exit 0
fi
[ "$#" = 0 ] || fail "未知参数。双击正常上传；终端使用 --check 检查连接。"

if ! mkdir .github-upload.lock 2>/dev/null; then
    lock_pid=$(cat .github-upload.lock/pid 2>/dev/null || true)
    if [[ "$lock_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
        rm -f .github-upload.lock/pid
        rmdir .github-upload.lock 2>/dev/null || true
        mkdir .github-upload.lock 2>/dev/null || fail "已有另一个上传任务，请等待它完成。"
    else
        fail "已有另一个上传任务，请等待它完成。如之前异常退出，请删除本目录的隐藏文件夹 .github-upload.lock 后重试。"
    fi
fi
LOCK_OWNED=1
printf '%s\n' "$$" > .github-upload.lock/pid

step 2 "准备公开仓库"
root=$(git rev-parse --show-toplevel 2>/dev/null || true)
[ -z "$root" ] || [ "$root" = "$PROJECT_DIR" ] || fail "本目录属于上层 Git 仓库，请将算子文件夹移到独立目录后运行。"
ORIGIN=""
if [ -n "$root" ]; then
    ORIGIN=$(git config --get remote.origin.url 2>/dev/null || true)
fi
if [ -n "$ORIGIN" ]; then
    case "$ORIGIN" in
        https://github.com/*) REPO=${ORIGIN#https://github.com/} ;;
        git@github.com:*) REPO=${ORIGIN#git@github.com:} ;;
        *) fail "origin 不是 github.com 仓库，请先检查本目录的 Git 远程配置。" ;;
    esac
    REPO=${REPO%.git}
    [[ "$REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || fail "GitHub 仓库地址格式异常，请检查 origin。"
else
    repo_name=$(ask "首次上传 · 创建公开仓库" "GitHub 账号：$LOGIN
本目录的文件将上传到公开仓库。请输入仓库名：" "$DEFAULT_REPO" "继续") || exit 0
    [[ "$repo_name" =~ ^[A-Za-z0-9_.-]+$ ]] && [ "$repo_name" != . ] && [ "$repo_name" != .. ] || fail "仓库名请使用英文字母、数字、短横线、下划线或点。"
    REPO="$LOGIN/$repo_name"
fi
REPO_URL="https://github.com/$REPO"
NEEDS_CREATE=0
ERROR_DETAIL="无法检查公开仓库 ${REPO}，请确认网络和当前账号的访问权限。"
if visibility=$(gh repo view "$REPO" --json visibility --jq .visibility); then
    [ "$visibility" = PUBLIC ] || fail "$REPO 当前不是公开仓库。请先在 GitHub 修改可见性，或使用新的仓库名。"
elif [ -n "$ORIGIN" ]; then
    fail "无法访问已关联的 ${REPO}，请检查网络、仓库是否存在及账号权限。"
else
    NEEDS_CREATE=1
fi
printf '  仓库：%s\n' "$REPO_URL"

step 3 "保存本次修改"
ERROR_DETAIL="无法保存本地提交，请检查终端中的 Git 错误。"
if [ -z "$root" ]; then
    git init -b main
fi
BRANCH=$(git symbolic-ref --quiet --short HEAD) || fail "当前处于 detached HEAD，请先切换到要上传的分支。"
# 未设置作者时仅补全本仓库，使用 GitHub 提供的隐私邮箱。
if ! git var GIT_AUTHOR_IDENT >/dev/null 2>&1; then
    user_id=$(gh api user --jq .id)
    git config --local user.name "$LOGIN"
    git config --local user.email "$user_id+$LOGIN@users.noreply.github.com"
fi
if [ -n "$(git status --porcelain)" ]; then
    git -c core.quotePath=false status --short
    MESSAGE=$(ask "上传 CUDA 算子" "这次修改了什么？留空将自动填写时间。
目标：${REPO}（公开）" "" "上传") || exit 0
    [ -n "$MESSAGE" ] || MESSAGE="更新算子 $(date '+%Y-%m-%d %H:%M:%S')"
    git add --all
    if ! git diff --cached --quiet; then
        git commit -m "$MESSAGE"
    fi
fi
git rev-parse --verify HEAD >/dev/null 2>&1 || fail "目录中没有可上传的文件。请先放入算子代码。"

step 4 "上传到 GitHub"
if [ "$NEEDS_CREATE" = 1 ]; then
    ERROR_DETAIL="创建公开仓库 $REPO 失败。请检查网络、仓库名及账号权限，然后重试。"
    gh repo create "$REPO" --public --description "CUDA operators and GEMM experiments"
fi
ERROR_DETAIL="关联仓库失败，请检查本地 Git 配置。"
if [ -z "$ORIGIN" ]; then
    git remote add origin "$REPO_URL.git"
elif [ "$ORIGIN" != "$REPO_URL.git" ]; then
    # HTTPS 使用与 ShaneOS 相同的 gh 凭据，避免依赖 SSH 或全局代理。
    git remote set-url origin "$REPO_URL.git"
fi
ERROR_DETAIL="推送失败。请检查代理和 GitHub 权限；若提示 rejected / fetch first，说明远端有新提交，需要先合并后重试。"
# 普通 push：保留远端历史；没有新改动时仍会重试此前失败的提交。
git "${GIT_NETWORK[@]}" push --set-upstream origin "$BRANCH"
printf '\n✓ 上传完成\n  %s\n' "$REPO_URL"
osascript - "$REPO_URL" <<'APPLESCRIPT' >/dev/null 2>&1 || true
on run argv
    set resultDialog to display dialog ("当前版本已同步到公开 GitHub 仓库。" & return & return & item 1 of argv) buttons {"完成", "打开仓库"} default button "完成" with title "算子上传完成"
    if button returned of resultDialog is "打开仓库" then open location (item 1 of argv)
end run
APPLESCRIPT
