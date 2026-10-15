# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical RoboCasa full-24 task registry in training-mixture order."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Full24Task:
    task_id: int
    family_id: int
    label: str
    dataset_name: str
    env_name: str
    source: str
    destination: str
    group: str


def _task(
    task_id: int,
    stem: str,
    label: str,
    source: str,
    destination: str,
    group: str,
) -> Full24Task:
    suffix = "_GR1ArmsAndWaistFourierHands"
    return Full24Task(
        task_id=task_id,
        family_id=task_id,
        label=label,
        dataset_name=f"gr1_unified.{stem}{suffix}_1000",
        env_name=f"gr1_unified/{stem}{suffix}_Env",
        source=source,
        destination=destination,
        group=group,
    )


FULL24_TASKS = (
    _task(0, "PnPBottleToCabinetClose", "bottle", "bottle", "cabinet", "pnp6"),
    _task(1, "PnPCanToDrawerClose", "can", "can", "drawer", "pnp6"),
    _task(2, "PnPCupToDrawerClose", "cup", "cup", "drawer", "pnp6"),
    _task(3, "PnPMilkToMicrowaveClose", "milk", "milk", "microwave", "pnp6"),
    _task(
        4,
        "PnPPotatoToMicrowaveClose",
        "potato",
        "potato",
        "microwave",
        "pnp6",
    ),
    _task(5, "PnPWineToCabinetClose", "wine", "wine", "cabinet", "pnp6"),
    _task(
        6,
        "PosttrainPnPNovelFromCuttingboardToBasketSplitA",
        "cuttingboard_to_basket",
        "cuttingboard",
        "basket",
        "novel",
    ),
    _task(
        7,
        "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA",
        "cuttingboard_to_cardboardbox",
        "cuttingboard",
        "cardboardbox",
        "novel",
    ),
    _task(
        8,
        "PosttrainPnPNovelFromCuttingboardToPanSplitA",
        "cuttingboard_to_pan",
        "cuttingboard",
        "pan",
        "novel",
    ),
    _task(
        9,
        "PosttrainPnPNovelFromCuttingboardToPotSplitA",
        "cuttingboard_to_pot",
        "cuttingboard",
        "pot",
        "novel",
    ),
    _task(
        10,
        "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA",
        "cuttingboard_to_tieredbasket",
        "cuttingboard",
        "tieredbasket",
        "novel",
    ),
    _task(
        11,
        "PosttrainPnPNovelFromPlacematToBasketSplitA",
        "placemat_to_basket",
        "placemat",
        "basket",
        "novel",
    ),
    _task(
        12,
        "PosttrainPnPNovelFromPlacematToBowlSplitA",
        "placemat_to_bowl",
        "placemat",
        "bowl",
        "novel",
    ),
    _task(
        13,
        "PosttrainPnPNovelFromPlacematToPlateSplitA",
        "placemat_to_plate",
        "placemat",
        "plate",
        "novel",
    ),
    _task(
        14,
        "PosttrainPnPNovelFromPlacematToTieredshelfSplitA",
        "placemat_to_tieredshelf",
        "placemat",
        "tieredshelf",
        "novel",
    ),
    _task(
        15,
        "PosttrainPnPNovelFromPlateToBowlSplitA",
        "plate_to_bowl",
        "plate",
        "bowl",
        "novel",
    ),
    _task(
        16,
        "PosttrainPnPNovelFromPlateToCardboardboxSplitA",
        "plate_to_cardboardbox",
        "plate",
        "cardboardbox",
        "novel",
    ),
    _task(
        17,
        "PosttrainPnPNovelFromPlateToPanSplitA",
        "plate_to_pan",
        "plate",
        "pan",
        "novel",
    ),
    _task(
        18,
        "PosttrainPnPNovelFromPlateToPlateSplitA",
        "plate_to_plate",
        "plate",
        "plate",
        "novel",
    ),
    _task(
        19,
        "PosttrainPnPNovelFromTrayToCardboardboxSplitA",
        "tray_to_cardboardbox",
        "tray",
        "cardboardbox",
        "novel",
    ),
    _task(
        20,
        "PosttrainPnPNovelFromTrayToPlateSplitA",
        "tray_to_plate",
        "tray",
        "plate",
        "novel",
    ),
    _task(
        21,
        "PosttrainPnPNovelFromTrayToPotSplitA",
        "tray_to_pot",
        "tray",
        "pot",
        "novel",
    ),
    _task(
        22,
        "PosttrainPnPNovelFromTrayToTieredbasketSplitA",
        "tray_to_tieredbasket",
        "tray",
        "tieredbasket",
        "novel",
    ),
    _task(
        23,
        "PosttrainPnPNovelFromTrayToTieredshelfSplitA",
        "tray_to_tieredshelf",
        "tray",
        "tieredshelf",
        "novel",
    ),
)

FULL24_DATASET_TO_TASK_ID = {
    task.dataset_name: task.task_id for task in FULL24_TASKS
}
FULL24_ENV_TO_TASK_ID = {task.env_name: task.task_id for task in FULL24_TASKS}

