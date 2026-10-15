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
# 基础镜像怎么选（背景见 docs/roles/README.md「硬约束」）：
#   客服已确认**没有可下载的官方基础镜像 tar 包**。官方口径里的「基础镜像」就是
#   咪咕仝学平台的**公有镜像**——Ubuntu 那条的说明原文写着「集成了通用软件、
#   tensorboard、jupyter，并支持用户基于此基础镜像定制自己的业务镜像」。
#   平台实测可选的公有镜像：vLLM / Ollama / LMDeploy / PyTorch×7 / MiniConda3×4 /
#   Ubuntu / PaddlePaddle / TensorFlow / DeepSeek / MiniMax / Qwen-Image ……
#
#   首选 **PyTorch 或 vLLM 公有镜像**（python + torch 已经装好，省掉几个 GB），
#   其次 Ubuntu（此时用 --build-arg INSTALL_DEPS=1 自己装）。
#
# 离线自检（必经）：
#   docker run --rm --network none ogc-participant:latest bash /opt/offline_check.sh

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

# INSTALL_DEPS=1 时自己装 python 依赖；基础镜像已带 torch 就保持 0。
# cu130 是为了 Blackwell(sm_120)（我们自己的笔记本）；平台是 RTX 4090，cu128 也够。
ARG INSTALL_DEPS=0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/root/.cache/huggingface \
    PARTICIPANT_CONTENT_ROOT=/root \
    PARTICIPANT_PROMPT_DIR=/root/participant/prompts

SHELL ["/bin/bash", "-lc"]

# ---------- 依赖（基础镜像已带 python+torch 时跳过） ----------
RUN if [ "$INSTALL_DEPS" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates \
      && rm -rf /var/lib/apt/lists/* \
      && python3 -m pip install --no-cache-dir --upgrade pip \
      && python3 -m pip install --no-cache-dir torch --index-url "${TORCH_INDEX}" \
      && python3 -m pip install --no-cache-dir transformers accelerate safetensors; \
    fi

# ---------- 参赛工具（= deploy/participant 的内容 + 队友的提示词） ----------
COPY participant/ /root/participant/

# ---------- 模型权重：必须打进镜像，运行时不许联网下载 ----------
# 本地先跑 python tools/fetch_model.py 下到 GivenData/participant/model/。
# Qwen2.5-3B fp16 约 6.2 GB；整个镜像要守住平台的 50 GB 上限。
COPY model/ /root/participant/model/

# ---------- 角色④⑤的检索与组包模块（缺失也能跑，wiring.py 会回退） ----------
COPY rag/ /root/rag/

# ---------- 自检脚本 ----------
COPY offline_check.sh /opt/offline_check.sh
RUN chmod +x /root/participant/start.sh /opt/offline_check.sh 2>/dev/null || true

# 故意不写 CMD/ENTRYPOINT：组委会按官方约定的命令启动，不是按我们的默认命令。
WORKDIR /root
