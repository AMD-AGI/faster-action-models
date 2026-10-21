#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Python environments for the DiT4DiT launchers (scripts/eval_dit4dit_libero.sh, scripts/eval_dit4dit_robocasa.sh)
# on AMD ROCm. DiT4DiT/requirements.txt pins CUDA PyTorch and nvidia-* wheels, so it cannot be installed as is.
# Run this inside the evaluation image (docker/Dockerfile.eval: ROCm 7.14, torch 2.11.0, Python 3.12, LIBERO),
# with OUT_DIR on a mounted path, and run the launchers in the same image:
#
#   docker run --rm -it --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER="$(id -un)" -v "$PWD:$PWD" -w "$PWD" \
#     fam-eval:rocm7.14 bash DiT4DiT/environments/setup_dit4dit_rocm_envs.sh "$PWD/envs"
# (or: bash scripts/in_eval_image.sh bash DiT4DiT/environments/setup_dit4dit_rocm_envs.sh "$PWD/envs")
#
# It creates, under OUT_DIR:
#   model/      DIT4DIT_MODEL_PYTHON: requirements-model-rocm.txt (no torch, triton, nvidia-*) installed --no-deps
#               plus DiT4DiT itself, on top of the image's ROCm PyTorch (a .pth file adds the image's site-packages)
#   pyoverlay/  DIT4DIT_PYOVERLAY (docs/side-by-side-source-trees.md)
#   robocasa/   ROBOCASA_PYTHON: numpy 1.26.4, mujoco 3.2.6, scipy 1.15.3, gymnasium 1.x, pydantic 2.10.6 (the full-24
#               client's base_config.py) and RoboCasa-GR1, and
#   robosuite151-site-packages/  ROBOSUITE_OVERLAY (build_robosuite151_overlay.sh); both only when the
#               RoboCasa-GR1 checkout exists: bash DiT4DiT/environments/setup_robocasa_gr1.sh first
#   mujoco332/  INSTINCT_SIM_MUJOCO_PATH: mujoco 3.3.2 for RoboCasa with --sim instinct
# The LIBERO client (DIT4DIT_LIBERO_PYTHON) is the image's own python3.
# The environments use the image's site-packages, so they work only inside that image.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D4="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${1:-}"
[[ -n "$OUT" && "$OUT" != -h && "$OUT" != --help ]] || {
  sed -n '4,/^set -Eeuo/{/^set -Eeuo/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"
  echo "usage: bash DiT4DiT/environments/setup_dit4dit_rocm_envs.sh OUT_DIR"
  [[ -n "$OUT" ]] && exit 0 || exit 2
}
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
ROBOCASA_ROOT="${ROBOCASA_ROOT:-${D4}/robocasa-gr1-tabletop-tasks}"
BASE_PY="${BASE_PY:-python3}"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
# getpass.getuser() (torch, when DeepSpeed builds) reads USER before the passwd database, which a container started
# with --user for a host uid usually lacks.
export USER="${USER:-$(id -un 2>/dev/null || echo "u$(id -u)")}"

base_site="$("$BASE_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
base_torch="$("$BASE_PY" -c 'import torch; print(torch.__version__)')"
echo "base python: $(command -v "$BASE_PY"), site-packages ${base_site}, torch ${base_torch}"

mkvenv() {  # a venv that also sees the image's site-packages (after its own)
  "$BASE_PY" -m venv "$1" 2>/dev/null || { "$BASE_PY" -m venv --without-pip "$1" && "$1/bin/python" -m ensurepip -q; }
  local site; site="$("$1/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  echo "$base_site" > "$site/zz_base_image_site_packages.pth"
}

echo "== model environment: $OUT/model"
mkvenv "$OUT/model"
"$OUT/model/bin/pip" install -q --no-deps -r "${SCRIPT_DIR}/requirements-model-rocm.txt"
"$OUT/model/bin/pip" install -q --no-deps -e "$D4"
# setuptools inside the venv itself: deepspeed (imported by DiT4DiT's trainer modules) imports distutils, which Python
# 3.12 no longer ships; only a setuptools installed in the venv activates its distutils shim (the image's copy is seen
# but its .pth is not run). The DeepSpeed-launched policy trainers still need the training environment of
# docs/cae-dit4dit-robocasa.md (rocm/primus); this venv serves the launchers and the action-VAE trainer.
"$OUT/model/bin/pip" install -q --no-deps --ignore-installed setuptools==81.0.0

echo "== pyoverlay: $OUT/pyoverlay"
"$OUT/model/bin/python" -m pip install -q --target "$OUT/pyoverlay" anyio certifi docstring_parser h11 httpcore \
  httpx idna typeguard typing_extensions tyro

echo "== mujoco 3.3.2: $OUT/mujoco332"
"$BASE_PY" -m pip install -q --no-deps --target "$OUT/mujoco332" mujoco==3.3.2

if [[ -d "$ROBOCASA_ROOT/robocasa" ]]; then
  echo "== RoboCasa client environment: $OUT/robocasa"
  mkvenv "$OUT/robocasa"
  # scipy: the image's scipy is built for NumPy 2; this venv pins NumPy 1.26.4
  "$OUT/robocasa/bin/pip" install -q numpy==1.26.4 mujoco==3.2.6 scipy==1.15.3 'gymnasium>=1,<2' pydantic==2.10.6 tyro websockets msgpack
  printf 'numpy==1.26.4\nmujoco==3.2.6\nscipy==1.15.3\n' > "$OUT/robocasa_constraints.txt"
  "$OUT/robocasa/bin/pip" install -q -c "$OUT/robocasa_constraints.txt" -e "$ROBOCASA_ROOT"
  echo "== robosuite 1.5.1 overlay: $OUT/robosuite151-site-packages"
  PYTHON_BIN="$OUT/robocasa/bin/python" bash "${SCRIPT_DIR}/build_robosuite151_overlay.sh" \
    "$OUT/robosuite151-site-packages" > "$OUT/robosuite151-overlay.log"
else
  echo "== RoboCasa client environment skipped: no RoboCasa-GR1 checkout at $ROBOCASA_ROOT"
  echo "   (bash DiT4DiT/environments/setup_robocasa_gr1.sh, then rerun this script)"
fi

echo "== checks"
(cd "$D4" && "$OUT/model/bin/python" -c '
import distutils.ccompiler, sympy, torch, torch._inductor.compile_fx, diffusers, transformers, tokenizers, numpy, DiT4DiT.model.framework
print("model env ok: torch", torch.__version__, "sympy", sympy.__version__, "diffusers", diffusers.__version__,
      "transformers", transformers.__version__, "tokenizers", tokenizers.__version__, "numpy", numpy.__version__)')
"$OUT/model/bin/python" -c "import torch; assert torch.__version__ == '${base_torch}', torch.__version__"
if [[ -x "$OUT/robocasa/bin/python" ]]; then
  # the launcher puts the robosuite 1.5.1 overlay first on PYTHONPATH; without it the image's robosuite 1.4.0 is found
  (cd "$D4" && PYTHONPATH="$OUT/robosuite151-site-packages:$ROBOCASA_ROOT:$D4" "$OUT/robocasa/bin/python" -c '
import gymnasium, mujoco, numpy, scipy.spatial, robosuite, robocasa, pydantic, websockets, msgpack
from robocasa.utils.gym_utils import GrootRoboCasaEnv
print("robocasa env ok: gymnasium", gymnasium.__version__, "mujoco", mujoco.__version__, "numpy", numpy.__version__,
      "scipy", scipy.__version__, "robosuite", robosuite.__version__)')
fi
PYTHONPATH="$OUT/mujoco332" "$BASE_PY" -c 'import mujoco; assert mujoco.__version__ == "3.3.2", mujoco.__version__'

cat <<EOF

Export (the launchers run inside the same image: bash scripts/in_eval_image.sh bash scripts/eval_dit4dit_*.sh ...):
  export DIT4DIT_MODEL_PYTHON=$OUT/model/bin/python
  export DIT4DIT_PYOVERLAY=$OUT/pyoverlay
  export DIT4DIT_LIBERO_PYTHON=$(command -v "$BASE_PY")
  export INSTINCT_SIM_MUJOCO_PATH=$OUT/mujoco332
EOF
[[ -x "$OUT/robocasa/bin/python" ]] && echo "  export ROBOCASA_PYTHON=$OUT/robocasa/bin/python ROBOCASA_ROOT=$ROBOCASA_ROOT" \
  && echo "  export ROBOSUITE_OVERLAY=$OUT/robosuite151-site-packages"
exit 0
