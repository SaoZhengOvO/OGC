# 参赛镜像。
#
# ⚠️ 这个 Dockerfile 的构建上下文**不是仓库根目录**，而是 `deploy/build.sh`
#    组装出来的暂存目录 `deploy/dist/context/`。原因是 `rag/` 与 `prompts/`
#    由队友（角色②③④⑤）产出，可能还不存在，而 docker 的 COPY 遇到不存在的
#    源会直接构建失败——build.sh 会把缺的目录补成空目录，保证镜像随时可建。
#
#    手工构建时请先跑一次：bash deploy/build.sh --stage-only
#
# 组委会唯一会执行的是：
#     bash /root/participant/start.sh TEST_FILE RESULT_DIR
#
# ---------------------------------------------------------------------------
# 2026-10-17 修正（P0）：这个镜像之前会「构建成功、自检通过、组委会一跑就崩」。
#
#   原因：INSTALL_DEPS=0（默认值）时一个包都不装，而 run_inference.py:357 的
#   `import torch` 是**惰性导入**，藏在 `backend == "transformers"` 分支里——
#   于是 build 能过、offline_check.sh 也能过，直到真跑推理才 ModuleNotFoundError。
#
#   修法两条，缺一不可：
#     1. 依赖拆两半：torch 可选（基座镜像自带就不装），
#        transformers / accelerate / safetensors 一律装。
#     2. 构建末尾做**硬断言**：四个包任何一个 import 不进来，就让构建失败。
#        宁可现在红，不要在组委会那里红。
# ---------------------------------------------------------------------------
#
# 基础镜像怎么选（背景见 docs/roles/README.md「硬约束」）：
#   客服已确认**没有可下载的官方基础镜像 tar 包**。官方口径里的「基础镜像」就是
#   咪咕仝学平台的**公有镜像**——Ubuntu 那条的说明原文写着「集成了通用软件、
#   tensorboard、jupyter，并支持用户基于此基础镜像定制自己的业务镜像」。
#   平台实测可选的公有镜像：vLLM / Ollama / LMDeploy / PyTorch×7 / MiniConda3×4 /
#   Ubuntu / PaddlePaddle / TensorFlow / DeepSeek / MiniMax / Qwen-Image ……
#
#   **默认走自建路线**（ubuntu:22.04 + 自己装依赖），理由是版本可控、不依赖平台
#   私有 registry 的 pull 权限。如果构建机能 pull 到平台公有镜像，可以改成：
#     BASE_IMAGE=<平台 PyTorch 镜像> INSTALL_TORCH=0 bash deploy/build.sh
#
# 离线自检（必经）：
#   docker run --rm --network none ogc-participant:latest bash /opt/offline_check.sh

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

# INSTALL_TORCH=1：自己装 torch（ubuntu 路线）
# INSTALL_TORCH=0：基座镜像已带 torch，只补 transformers 那一层
ARG INSTALL_TORCH=1

# CUDA 轮子版本。默认 cu124 —— 它对驱动的版本要求最宽（NVIDIA >= 525.60.13），
# RTX 4090 无论平台装的是 525 / 550 / 560 驱动都能跑。
# 不要随手改成 cu128 / cu130：那两个要求 570+ / 580+ 驱动；跑不起来时
# torch.cuda.is_available() 会静默变 False，推理退化到 CPU，直接超时。
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

# pip 源，构建机网络不好时可以覆盖成国内镜像
ARG PIP_INDEX_URL=https://pypi.org/simple

# 兼容旧参数：INSTALL_DEPS 沿用原有语义（1 = 连 torch 一起装）
ARG INSTALL_DEPS=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/root/.cache/huggingface \
    PARTICIPANT_CONTENT_ROOT=/root \
    PARTICIPANT_PROMPT_DIR=/root/participant/prompts

SHELL ["/bin/bash", "-lc"]

# ---------- 1/3 Python 运行时 ----------
# 基座镜像可能连 python3 都没有（ubuntu:22.04 自带 3.10，但不带 pip），
# 所以按「缺什么补什么」处理，而不是按固定的开关走。
RUN set -eux; \
    if ! command -v python3 >/dev/null 2>&1; then \
      apt-get update; \
      apt-get install -y --no-install-recommends python3 python3-pip ca-certificates; \
      rm -rf /var/lib/apt/lists/*; \
    fi; \
    if ! python3 -m pip --version >/dev/null 2>&1; then \
      apt-get update; \
      apt-get install -y --no-install-recommends python3-pip ca-certificates; \
      rm -rf /var/lib/apt/lists/*; \
    fi; \
    python3 --version; \
    python3 -m pip --version

# ---------- 2/3 推理依赖 ----------
# torch 可选；transformers 那一层**永远装**。
# 这一层是 P0 的正面修复：以前 INSTALL_DEPS=0 时连 transformers 都没装。
RUN set -eux; \
    if [ "${INSTALL_DEPS}" = "1" ]; then INSTALL_TORCH=1; fi; \
    python3 -m pip install --no-cache-dir --upgrade pip; \
    if [ "${INSTALL_TORCH}" = "1" ]; then \
      python3 -m pip install --no-cache-dir torch --index-url "${TORCH_INDEX}"; \
    fi; \
    python3 -m pip install --no-cache-dir \
      --index-url "${PIP_INDEX_URL}" \
      "transformers>=4.40" \
      "accelerate>=0.29" \
      "safetensors>=0.4" \
      "sentencepiece>=0.1.99" \
      "protobuf>=4.21"

# ---------- 2.5/3 硬断言：缺依赖就让构建失败 ----------
# 这一段是整个 Dockerfile 里最重要的几行。
# 它把「镜像到底能不能跑」从「组委会那一刻才知道」提前到「build 那一刻就知道」。
RUN set -eux; \
    python3 -c "import torch, transformers, accelerate, safetensors; \
print('[deps] torch        ', torch.__version__); \
print('[deps] transformers ', transformers.__version__); \
print('[deps] accelerate   ', accelerate.__version__); \
print('[deps] safetensors  ', safetensors.__version__); \
print('[deps] torch.cuda.is_available() =', torch.cuda.is_available(), '(构建机通常无 GPU，False 属正常)')"

# ---------- 3/3 参赛工具 ----------
# = deploy/participant 的内容（含 build.sh 暂存进来的 prompts/）
COPY participant/ /root/participant/

# 模型权重：必须打进镜像，运行时不许联网下载。
# run_inference.py:142 用 `ROOT / model_path` 解析，ROOT = /root/participant，
# 所以 "./model" 命中 /root/participant/model —— 与启动时的 CWD 无关。
COPY model/ /root/participant/model/

# 角色④⑤的检索与组包模块（缺失也能跑，wiring.py 会回退）。
# wiring.py:58-70 会把 PARTICIPANT_CONTENT_ROOT（=/root）插进 sys.path，
# 所以 /root/rag 能被 import。
COPY rag/ /root/rag/

# 自检脚本
COPY offline_check.sh /opt/offline_check.sh
# 不吞 stderr、不跟 `|| true`：这两个脚本是镜像里唯一的可执行入口，chmod 失败说明
# 前面的 COPY 就已经不对了，想在构建期炸，不想等平台启动时才报 Permission denied。
RUN chmod +x /root/participant/start.sh /opt/offline_check.sh

# 专家评审材料。官方硬性要求（GivenData/README.md:33-36）：技术方案、实验设计、报告、
# 平台运行记录要「打包，放到提交的镜像里，存放路径为 /root 下，命名为
# AI技术赛道-数字人综合情感对话模型-<队名>」。
# build.sh 已把该目录名带上队名建在 review/ 下，这里整目录落到 /root 即可
# —— 这样队名是构建期决定的，Dockerfile 不用跟着改。
COPY review/ /root/

# 故意不写 CMD/ENTRYPOINT：组委会按官方约定的命令启动，不是按我们的默认命令。
WORKDIR /root
