#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_DIR=${1:-"${SCRIPT_DIR}/robosuite151-site-packages"}

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" -m pip install \
  --target "${OUTPUT_DIR}" \
  --no-deps \
  --requirement "${SCRIPT_DIR}/requirements-eval-overlay.txt"

"${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
distributions = {
    distribution.metadata["Name"].lower(): distribution
    for distribution in importlib.metadata.distributions(path=[str(root)])
}
robosuite_record = next(root.glob("robosuite-*.dist-info/RECORD"))
manifest = {
    "overlay": str(root),
    "versions": {
        name: distribution.version
        for name, distribution in distributions.items()
        if name in {"numpy", "mujoco", "robosuite"}
    },
    "robosuite_record_sha256": hashlib.sha256(
        robosuite_record.read_bytes()
    ).hexdigest(),
}
(root / "ROBOSUITE_OVERLAY.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY
