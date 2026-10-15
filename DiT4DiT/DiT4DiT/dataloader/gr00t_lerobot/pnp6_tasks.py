# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical RoboCasa PnP-6 task registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PnP6Task:
    task_id: int
    label: str
    receptacle: str
    dataset_name: str
    env_name: str


PNP6_TASKS = (
    PnP6Task(
        0,
        "bottle",
        "cabinet",
        "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
        "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    ),
    PnP6Task(
        1,
        "can",
        "drawer",
        "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
        "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    ),
    PnP6Task(
        2,
        "cup",
        "drawer",
        "gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
        "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    ),
    PnP6Task(
        3,
        "milk",
        "microwave",
        "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
        "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    ),
    PnP6Task(
        4,
        "potato",
        "microwave",
        "gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
        "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    ),
    PnP6Task(
        5,
        "wine",
        "cabinet",
        "gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
        "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    ),
)

PNP6_DATASET_TO_TASK_ID = {task.dataset_name: task.task_id for task in PNP6_TASKS}
PNP6_ENV_TO_TASK_ID = {task.env_name: task.task_id for task in PNP6_TASKS}
PNP6_LABEL_TO_TASK_ID = {task.label: task.task_id for task in PNP6_TASKS}


def pnp6_task_id_for_dataset(dataset_name: str) -> int:
    try:
        return PNP6_DATASET_TO_TASK_ID[dataset_name]
    except KeyError as exc:
        raise KeyError(f"Dataset is not in the PnP-6 task registry: {dataset_name}") from exc


def pnp6_task_id_for_env(env_name: str) -> int:
    try:
        return PNP6_ENV_TO_TASK_ID[env_name]
    except KeyError as exc:
        raise KeyError(f"Environment is not in the PnP-6 task registry: {env_name}") from exc
