
# -----------------------------------------------------------------------
# Base: CUDA 12.6.2 + cuDNN on Ubuntu 22.04
# -----------------------------------------------------------------------
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ARG TZ
ENV TZ="${TZ:-UTC}"
ENV DEBIAN_FRONTEND=noninteractive

# MuJoCo headless rendering via EGL (no display server needed in Docker).
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

# Keep Python output unbuffered so logs appear immediately.
ENV PYTHONUNBUFFERED=1

ARG CLAUDE_CODE_VERSION=latest

# -----------------------------------------------------------------------
# System dependencies + Python 3.10
# Ubuntu 22.04 ships Python 3.10
# -----------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.10
    python3 python3-dev python3-venv python3-pip \
    # Build tools
    git wget curl build-essential cmake \
    # OpenGL / EGL for MuJoCo offscreen rendering
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    # Required by mujoco / robosuite
    patchelf \
    libxrandr2 libxinerama1 libxcursor1 libxi6 \
    # FFmpeg for video saving
    ffmpeg \
    # CLI / networking tools
    less \
    procps \
    fzf \
    zsh \
    man-db \
    unzip \
    gnupg2 \
    gh \
    iptables \
    ipset \
    iproute2 \
    dnsutils \
    aggregate \
    jq \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 / python3 / python all resolve to 3.10.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 10 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 10

# Bootstrap pip for 3.10 (deadsnakes does not ship pip for 3.10).
RUN python3 -m pip install --upgrade pip setuptools wheel

# ── Node.js 20 ─────────────────────────────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Source is bind-mounted at runtime; /source is the working directory.
WORKDIR /source

# -----------------------------------------------------------------------
# Python dependencies
# Copy only requirements.txt for better layer caching; the rest of the
# source tree is provided via a read-only bind mount at runtime.
# -----------------------------------------------------------------------
COPY requirements*.txt /tmp/
# flash-attn requires --no-build-isolation (needs torch headers at build time).
# Strip it from requirements.txt, install the rest, then install flash-attn separately.
RUN python3 -m pip install --upgrade setuptools && \
    grep -v "flash[-_]attn" /tmp/requirements.txt > /tmp/requirements_noflash.txt && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements_noflash.txt -r /tmp/requirements-libero.txt && \
    python3 -m pip install --no-cache-dir flash-attn --no-build-isolation

# -----------------------------------------------------------------------
# LIBERO benchmark
# Clone to a permanent path so the source tree is always present.
# pip install pulls in all dependencies (robosuite, bddl, …); the
# package itself is made importable at runtime via PYTHONPATH because
# LIBERO's pyproject.toml doesn't describe its layout for pip.
# -----------------------------------------------------------------------
RUN git clone --depth 1 \
    https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO \
    && python3 -m pip install --no-cache-dir \
           "robosuite==1.4.1" bddl h5py lxml egl_probe gym \
           "tensorflow==2.15.0" tensorflow-datasets tensorflow-graphics \
    && python3 -m pip install --no-cache-dir --no-deps /opt/LIBERO
ENV PYTHONPATH="/opt/LIBERO"

RUN pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla

# LIBERO's libero/libero/__init__.py runs an interactive setup wizard at
# import time (calls input() to ask for the dataset root directory).
# Pre-run it here with input() patched so it writes its config file and
# all subsequent imports in the container are non-interactive.
RUN mkdir -p /opt/LIBERO/datasets /opt/LIBERO/libero/datasets \
    && python3 - <<'EOF'
import builtins
builtins.input = lambda *a, **kw: "/opt/LIBERO/datasets"
from libero.libero import benchmark
print("LIBERO init OK")
EOF

# -----------------------------------------------------------------------
# NOTE: OpenVLA is NOT installed as a package.
# The model code is fetched automatically by transformers via
# trust_remote_code=True when AutoModelForVision2Seq.from_pretrained() runs.
# -----------------------------------------------------------------------

# ── git-delta ──────────────────────────────────────────────────────────────
ARG GIT_DELTA_VERSION=0.18.2
RUN ARCH=$(dpkg --print-architecture) \
    && wget -q "https://github.com/dandavison/delta/releases/download/${GIT_DELTA_VERSION}/git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" \
    && dpkg -i "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb" \
    && rm "git-delta_${GIT_DELTA_VERSION}_${ARCH}.deb"

# ── zsh ────────────────────────────────────────────────────────────────────
ENV SHELL=/bin/zsh
ENV EDITOR=vim
ENV VISUAL=vim

ARG ZSH_IN_DOCKER_VERSION=1.2.0
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v${ZSH_IN_DOCKER_VERSION}/zsh-in-docker.sh)" -- \
    -p git \
    -p fzf \
    -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
    -a "source /usr/share/doc/fzf/examples/completion.zsh" \
    -x

# ── Claude Code ────────────────────────────────────────────────────────────
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# ── Firewall + entrypoint ──────────────────────────────────────────────────
COPY init-firewall.sh /usr/local/bin/init-firewall.sh
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/init-firewall.sh /usr/local/bin/entrypoint.sh

# ── Environment ────────────────────────────────────────────────────────────
ENV PYTHONPATH="/source:/source/src:/opt/LIBERO"
ENV HF_HOME=/hf_cache
ENV NODE_OPTIONS=--max-old-space-size=4096

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "eval_models.run_eval", "--help"]
