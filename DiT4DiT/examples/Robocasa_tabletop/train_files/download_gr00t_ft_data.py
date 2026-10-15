# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
import re
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import HfHubHTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
from pathlib import Path


REPO_ID = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
REPO_TYPE = "dataset"
REVISION = "ea7ac0b68f87da62f1e726771bba0fe74300802f"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
)

PNP_6_FOLDERS = [
    "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
]

PNP_3_FOLDERS = [
    "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
]

FOLDERS = PNP_6_FOLDERS + [
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected RoboCasa GR00T-X task folders.")
    parser.add_argument(
        "--folder-group",
        choices=("all", "pnp6", "pnp3"),
        default="all",
        help="Download all 24 folders, six standard PnP folders, or the selected three-task subset.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Parallel downloads. Keep this low to avoid Hugging Face API rate limits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination root for the selected dataset folders.",
    )
    return parser.parse_args()


def retry_wait_time(exc: Exception, attempt: int) -> float:
    is_rate_limited = (
        isinstance(exc, HfHubHTTPError)
        and exc.response is not None
        and exc.response.status_code == 429
    )
    if is_rate_limited:
        rate_limit = exc.response.headers.get("RateLimit", "")
        reset_match = re.search(r"t=(\d+)", rate_limit)
        reset_seconds = int(reset_match.group(1)) if reset_match else 300
        return reset_seconds + random.uniform(5.0, 15.0)
    return min(60.0, random.uniform(1.0, 5.0) * attempt)


def list_folder_files(api: HfApi, folder: str, max_retries: int = 10) -> list[str]:
    for attempt in range(1, max_retries + 1):
        try:
            entries = api.list_repo_tree(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                revision=REVISION,
                path_in_repo=folder,
                recursive=True,
            )
            return [entry.path for entry in entries if isinstance(entry, RepoFile)]
        except Exception as exc:
            wait_time = retry_wait_time(exc, attempt)
            print(
                f"[{attempt}/{max_retries}] Could not list {folder} ({exc}); "
                f"retrying in {wait_time:.1f}s",
                flush=True,
            )
            time.sleep(wait_time)

    raise RuntimeError(f"Giving up after {max_retries} attempts to list {folder}")


def download_with_retry(
    filename: str,
    local_dir: Path,
    max_retries: int = 50,
) -> bool:
    """
    Download a single file from Hugging Face Hub with retry logic.

    Args:
        filename: Path of the file in the repository.
        max_retries: Maximum number of retry attempts.

    Returns:
        True if download succeeds, False otherwise.
    """
    if (local_dir / filename).is_file():
        return True

    for attempt in range(1, max_retries + 1):
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                revision=REVISION,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            return True
        except Exception as exc:
            wait_time = retry_wait_time(exc, attempt)
            print(
                f"[{attempt}/{max_retries}] "
                f"Download failed: {filename} ({exc}); "
                f"retrying in {wait_time:.1f}s"
            )
            time.sleep(wait_time)

    print(f"Giving up after {max_retries} retries: {filename}")
    return False


def main() -> None:
    args = parse_args()
    folder_groups = {
        "all": FOLDERS,
        "pnp6": PNP_6_FOLDERS,
        "pnp3": PNP_3_FOLDERS,
    }
    folders = folder_groups[args.folder_group]

    print(f"Listing files for {len(folders)} selected folders...", flush=True)
    api = HfApi()
    target_files: list[str] = []
    for folder in folders:
        folder_files = list_folder_files(api, folder)
        target_files.extend(folder_files)
        print(f"  {folder}: {len(folder_files)} files", flush=True)

    print(
        f"Found {len(target_files)} matching files across "
        f"{len(folders)} '{args.folder_group}' folders.\n",
        flush=True,
    )

    max_workers = args.max_workers
    print(f"Starting parallel download with {max_workers} workers...\n")

    failed_files: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_with_retry, filename, args.output_dir): filename
            for filename in target_files
        }

        for idx, future in enumerate(as_completed(futures), start=1):
            filename = futures[future]
            try:
                success = future.result()
                if success:
                    if idx % 100 == 0 or idx == len(target_files):
                        print(
                            f"[{idx}/{len(target_files)}] Files ready; "
                            f"latest: {filename}",
                            flush=True,
                        )
                else:
                    failed_files.append(filename)
            except Exception as exc:
                print(f"Unexpected error for {filename}: {exc}")
                failed_files.append(filename)

    print("\nAll download attempts finished.")

    if failed_files:
        print(f"{len(failed_files)} files failed to download:")
        for filename in failed_files:
            print(f"  - {filename}")
    else:
        print("All files downloaded successfully.")


if __name__ == "__main__":
    main()
