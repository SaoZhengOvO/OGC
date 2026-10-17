#!/usr/bin/env bash
# 组装暂存构建上下文并打出参赛镜像。
#
# 为什么要有暂存目录：队友的 rag/ 与 prompts/ 可能还不存在，而 docker 的 COPY
# 遇到不存在的源会直接失败。本脚本把它们补成空目录，保证**任何时候都能构建**，
# 缺失的模块由 deploy/participant/wiring.py 回退到官方提示词。
#
# 用法：
#   bash deploy/build.sh                      # 组装 + 构建
#   bash deploy/build.sh --stage-only         # 只组装暂存目录（检查用）
#   BASE_IMAGE=<平台公有镜像> bash deploy/build.sh
#   bash deploy/build.sh --tar                # 额外导出 deploy/dist/ogc-participant.tar
#
# 环境要求：Linux / WSL（Windows 原生 Git Bash 的 cp -al 与 docker 路径映射容易出问题）。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY="$REPO_ROOT/deploy"
CONTEXT="$DEPLOY/dist/context"
IMAGE_TAG="${IMAGE_TAG:-ogc-participant:latest}"
BASE_IMAGE="${BASE_IMAGE:-ubuntu:22.04}"
# INSTALL_TORCH=1 自己装 torch（ubuntu 路线）；
# 基座镜像已带 torch 时设 0，只补 transformers 那一层。
INSTALL_TORCH="${INSTALL_TORCH:-1}"
# 兼容老用法：INSTALL_DEPS=1 等价于 INSTALL_TORCH=1
INSTALL_DEPS="${INSTALL_DEPS:-0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
MAKE_TAR=0
STAGE_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --tar) MAKE_TAR=1 ;;
    --stage-only) STAGE_ONLY=1 ;;
    *) echo "未知参数：$arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "0/5 检查前置条件"

if ! command -v docker >/dev/null 2>&1; then
  echo "找不到 docker。本地只做暂存检查的话，加 --stage-only。" >&2
  [ "$STAGE_ONLY" = "1" ] || exit 1
fi

MODEL_DIR="$REPO_ROOT/GivenData/participant/model"
if [ ! -d "$MODEL_DIR" ]; then
  cat >&2 <<'EOF'
模型权重目录不存在：GivenData/participant/model

先下载：
    python tools/fetch_model.py            # 默认 Qwen/Qwen2.5-3B-Instruct
    python tools/fetch_model.py --check    # 校验字节数是否完整

注意：官方推理脚本按 model_path="./model" 相对 /root/participant 解析，
所以权重必须放在 GivenData/participant/model/，不要改这个位置。
EOF
  exit 1
fi

if [ -f "$REPO_ROOT/tools/fetch_model.py" ] && command -v python3 >/dev/null 2>&1; then
  say "1/5 校验模型权重完整性（与远端比字节数）"
  python3 "$REPO_ROOT/tools/fetch_model.py" --check || {
    echo "权重不完整，重跑 python tools/fetch_model.py 续传。" >&2
    exit 1
  }
else
  say "1/5 跳过权重校验（没有 python3 或 fetch_model.py）"
fi

say "2/5 组装暂存上下文 -> $CONTEXT"
rm -rf "$CONTEXT"
mkdir -p "$CONTEXT/participant" "$CONTEXT/rag" "$CONTEXT/model"

# 参赛工具本体
cp -a "$DEPLOY/participant/." "$CONTEXT/participant/"

# 行尾兜底：把构建上下文里的文本文件统一成 LF。
#
# 为什么必须做：容器里跑的是 Linux，shell 脚本一旦带 CRLF，bash 会在第一行报
#   /bin/bash^M: bad interpreter: No such file or directory
# 而 start.sh 正是组委会唯一会执行的命令 —— 镜像直接判失败。
# .gitattributes 已经把仓库行尾钉成 LF，这里再兜一次，防止有人从
# GivenData/participant/（官方原件，本身就是 CRLF）直接拷了文件过来。
if command -v sed >/dev/null 2>&1; then
  crlf_count=0
  while IFS= read -r -d '' f; do
    if LC_ALL=C grep -q $'\r' "$f" 2>/dev/null; then
      sed -i 's/\r$//' "$f"
      crlf_count=$((crlf_count+1))
    fi
  done < <(find "$CONTEXT" -type f \
             \( -name '*.sh' -o -name '*.py' -o -name '*.json' -o -name '*.txt' \
                -o -name '*.md' -o -name '*.jinja' -o -name 'Dockerfile' \) -print0)
  if [ "$crlf_count" -gt 0 ]; then
    echo "  [已修正] $crlf_count 个文本文件带有 CRLF 行尾，已转成 LF"
  fi
fi

# 硬断言：任何 .sh 里都不能再残留 CR。宁可现在失败，也不要交一个跑不起来的镜像。
if LC_ALL=C grep -rlP '\r' "$CONTEXT"/participant/*.sh "$CONTEXT"/participant/**/*.sh 2>/dev/null | grep -q .; then
  echo "构建中止：暂存目录里仍有 .sh 脚本带 CRLF，进镜像会报 bad interpreter。" >&2
  LC_ALL=C grep -rlP '\r' "$CONTEXT"/participant/*.sh "$CONTEXT"/participant/**/*.sh 2>/dev/null | sed 's/^/    /' >&2
  exit 1
fi
echo "  行尾检查通过：所有 .sh 均为 LF"

# 队友的提示词（角色②③）；没有就留空目录，wiring.py 会回退到官方提示词
mkdir -p "$CONTEXT/participant/prompts"
if [ -d "$REPO_ROOT/prompts" ]; then
  cp -a "$REPO_ROOT/prompts/." "$CONTEXT/participant/prompts/" 2>/dev/null || true
fi
if [ -z "$(ls -A "$CONTEXT/participant/prompts" 2>/dev/null)" ]; then
  echo "  [注意] prompts/ 为空 —— 镜像会用官方内置提示词（可以跑，但拿不到队友的调优）。"
  : > "$CONTEXT/participant/prompts/.keep"
fi

# 队友的检索/组包模块（角色④⑤）；没有就留空，wiring.py 会回退
if [ -d "$REPO_ROOT/rag" ]; then
  # 语料是 .gitignore 里挡掉的构建产物（rag/corpus/corpus.jsonl，约 17 MB），
  # 从干净 clone 出来的机器上一定没有。这里不补建，镜像里的检索会静默退化成一堆空
  # —— 融合层照跑，只是每次都在拼一个空 retrieved 列表，看起来「生效了」其实没生效。
  if [ ! -s "$REPO_ROOT/rag/corpus/corpus.jsonl" ] \
     && [ -f "$REPO_ROOT/rag/corpus/build_corpus.py" ]; then
    if command -v python3 >/dev/null 2>&1; then
      echo "  构建检索语料（rag/corpus/corpus.jsonl 不存在）……"
      ( cd "$REPO_ROOT" && python3 -m rag.corpus.build_corpus ) || true
    elif command -v python >/dev/null 2>&1; then
      echo "  构建检索语料（用 python，不是 python3）……"
      ( cd "$REPO_ROOT" && python -m rag.corpus.build_corpus ) || true
    fi
  fi
  if [ ! -s "$REPO_ROOT/rag/corpus/corpus.jsonl" ]; then
    echo "  [警告] 检索语料仍然缺失 —— 镜像里的 RAG 会空转（融合层在、检索侧没有内容）。"
    echo "         先执行：python -m rag.corpus.build_corpus"
  else
    echo "  检索语料：$(wc -c <"$REPO_ROOT/rag/corpus/corpus.jsonl" | tr -d ' ') 字节"
  fi
  cp -a "$REPO_ROOT/rag/." "$CONTEXT/rag/" 2>/dev/null || true
  # __pycache__ 是本机的 Python 版本编译出来的 .pyc，进镜像既没用又脏
  # （容器里的 python 版本不同，这些 .pyc 一律会被忽略）。清掉。
  find "$CONTEXT/rag" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
  find "$CONTEXT/rag" -type f -name '*.pyc' -delete 2>/dev/null || true
fi
if [ -z "$(ls -A "$CONTEXT/rag" 2>/dev/null)" ]; then
  echo "  [注意] rag/ 为空 —— 镜像不带检索，组包走官方版（合规，但少了记忆融合）。"
  : > "$CONTEXT/rag/.keep"
fi

# 模型权重：优先硬链接（同盘瞬间完成、不额外占空间），失败退回完整复制
if cp -al "$MODEL_DIR/." "$CONTEXT/model/" 2>/dev/null; then
  echo "  权重用硬链接接入（不额外占磁盘）"
else
  echo "  硬链接不可用，改为完整复制（约 6.2 GB，会慢）"
  cp -a "$MODEL_DIR/." "$CONTEXT/model/"
fi

cp "$DEPLOY/Dockerfile" "$CONTEXT/Dockerfile"
cp "$DEPLOY/offline_check.sh" "$CONTEXT/offline_check.sh"

say "3/5 暂存内容清单"
du -sh "$CONTEXT"/* 2>/dev/null || true
echo "  参与文件数：$(find "$CONTEXT" -type f | wc -l)"

if [ "$STAGE_ONLY" = "1" ]; then
  say "只组装不构建（--stage-only），完成。"
  exit 0
fi

say "4/5 构建镜像 $IMAGE_TAG（BASE_IMAGE=$BASE_IMAGE, INSTALL_TORCH=$INSTALL_TORCH, INSTALL_DEPS=$INSTALL_DEPS）"
docker build \
  -f "$CONTEXT/Dockerfile" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "INSTALL_TORCH=$INSTALL_TORCH" \
  --build-arg "INSTALL_DEPS=$INSTALL_DEPS" \
  --build-arg "TORCH_INDEX=$TORCH_INDEX" \
  --build-arg "PIP_INDEX_URL=$PIP_INDEX_URL" \
  -t "$IMAGE_TAG" \
  "$CONTEXT"

say "5/5 离线自检（--network none，模拟组委会的断网环境）"
docker run --rm --network none "$IMAGE_TAG" bash /opt/offline_check.sh

if [ "$MAKE_TAR" = "1" ]; then
  say "额外：导出 tar 包 -> deploy/dist/ogc-participant.tar"
  docker save "$IMAGE_TAG" -o "$DEPLOY/dist/ogc-participant.tar"
  ls -lh "$DEPLOY/dist/ogc-participant.tar"
  echo "提醒：平台上限 50 GB（公有镜像 + 系统盘已用空间总和）。"
fi

say "完成。镜像：$IMAGE_TAG"
echo "本地按官方入口试跑："
echo "  docker run --rm --gpus all $IMAGE_TAG \\"
echo "    bash /root/participant/start.sh <TEST_FILE> <RESULT_DIR>"
