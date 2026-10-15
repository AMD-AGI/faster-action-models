# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Sourced by the DiT4DiT launchers; not executable on its own.
#
# build_diffusers_overlay MODEL_PYTHON ARM OVERLAY_DIR
#
# Builds a per-run copy of diffusers 3996788 (0.37.0.dev0) in OVERLAY_DIR,
# patched with src/dit4dit-geak/patches/ on the geak arm only, and sets
# `diffusers_overlay` to OVERLAY_DIR for the caller to put first on the server's
# PYTHONPATH. The model environment's own diffusers is never modified, so one
# environment serves every arm and a patched install cannot leak into the
# baseline.
#
# The source package is DIT4DIT_DIFFUSERS_SRC, or else the one the model
# environment resolves. It is located with find_spec rather than imported,
# because importing diffusers can pull in a broken flax.

build_diffusers_overlay() {
  local model_python="$1" arm="$2" overlay="$3"
  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

  local src="${DIT4DIT_DIFFUSERS_SRC:-}"
  if [[ -z "$src" ]]; then
    src="$(PYTHONPATH="${DIT4DIT_PYOVERLAY:-}${PYTHONPATH:+:${PYTHONPATH}}" "$model_python" -c '
import importlib.util, os
spec = importlib.util.find_spec("diffusers")
print(os.path.dirname(spec.origin) if spec and spec.origin else "")
')"
  fi
  [[ -f "${src}/__init__.py" ]] || {
    echo "no diffusers package found; set DIT4DIT_DIFFUSERS_SRC to a diffusers package directory" >&2
    return 2
  }
  grep -q '^__version__ = "0.37.0.dev0"$' "${src}/__init__.py" || {
    echo "diffusers at ${src} is not 0.37.0.dev0 (commit 3996788 is required)" >&2
    return 2
  }
  # The patches are against upstream 3996788. A copy that already carries a
  # GEAK edit would either reject the patch or, worse, hand the baseline arm
  # GEAK code.
  local -A upstream_md5=(
    [models/autoencoders/autoencoder_kl_wan.py]=da2de66f4d6b48995c815af84cbe07e1
    [models/transformers/transformer_cosmos.py]=88919a6ef7c70f6947ef2f1221128d29
  )
  local rel got
  for rel in "${!upstream_md5[@]}"; do
    got="$(md5sum "${src}/${rel}" | cut -d' ' -f1)"
    [[ "$got" == "${upstream_md5[$rel]}" ]] || {
      echo "diffusers ${rel} differs from upstream 3996788 (md5 ${got}); point" >&2
      echo "DIT4DIT_DIFFUSERS_SRC at an unmodified diffusers 3996788" >&2
      return 2
    }
  done

  rm -rf "$overlay"
  mkdir -p "$overlay"
  cp -a "$src" "${overlay}/diffusers" || return 1
  if [[ "$arm" == geak ]]; then
    local p
    for p in "${repo_root}/src/dit4dit-geak/patches/"*.patch; do
      patch -d "$overlay" -p1 --forward --batch --quiet <"$p" || return 1
    done
    local -A geak_md5=(
      [models/autoencoders/autoencoder_kl_wan.py]=999c1d6e1f9af86ea8865a7c3e472de8
      [models/transformers/transformer_cosmos.py]=5e91185f567dbd86fa371562edf9228b
    )
    for rel in "${!geak_md5[@]}"; do
      got="$(md5sum "${overlay}/diffusers/${rel}" | cut -d' ' -f1)"
      [[ "$got" == "${geak_md5[$rel]}" ]] || {
        echo "patched diffusers ${rel} has md5 ${got}, expected ${geak_md5[$rel]}" >&2
        return 1
      }
    done
  fi
  diffusers_overlay="$overlay"
}
