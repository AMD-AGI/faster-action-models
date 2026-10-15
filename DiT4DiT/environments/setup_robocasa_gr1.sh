#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DESTINATION=${1:-"${REPO_ROOT}/robocasa-gr1-tabletop-tasks"}
UPSTREAM=https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git
REVISION=4840e671596f93ca03651524b9f72ffb1aadfeff

if [[ ! -d "${DESTINATION}/.git" ]]; then
  git clone "${UPSTREAM}" "${DESTINATION}"
fi
git -C "${DESTINATION}" fetch origin "${REVISION}"
git -C "${DESTINATION}" checkout --detach "${REVISION}"
PATCH="${REPO_ROOT}/patches/robocasa-gr1-deterministic-reset.patch"
git -C "${DESTINATION}" apply --check "${PATCH}"
git -C "${DESTINATION}" apply "${PATCH}"

echo "Pinned and patched RoboCasa-GR1 at ${DESTINATION}"
echo "Install it with: python -m pip install -e ${DESTINATION}"
echo "Then download assets with:"
echo "  python ${DESTINATION}/robocasa/scripts/download_kitchen_assets.py"
