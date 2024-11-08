"""Script dividing the initial dataset into training and testing subsets."""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="divide_initial_dataset.log", format=FORMAT, level=logging.INFO)


def divide_initial_dataset():
    """This function divides the initial dataset into training and testing subsets."""
    color_subset_path = Path(
        "/Users/kubkodunaj/Desktop/jakub-dunaj-bachelors-thesis/datasets/initial_dataset/color"
    )
    segmented_subset_path = Path(
        "/Users/kubkodunaj/Desktop/jakub-dunaj-bachelors-thesis/datasets/initial_dataset/segmented"
    )
    artificial_background_subset_path = Path(
        "/Users/kubkodunaj/Desktop/jakub-dunaj-bachelors-thesis/datasets/initial_dataset/artificial_background"
    )
    final_dataset_path = Path(
        "/Users/kubkodunaj/Desktop/jakub-dunaj-bachelors-thesis/datasets/final_dataset"
    )

    subset_paths = [color_subset_path, segmented_subset_path, artificial_background_subset_path]
    existent_subset_paths = [folder.exists() for folder in subset_paths]

    for subset_path, exists in zip(subset_paths, existent_subset_paths, strict=False):
        if not exists or subset_path.parent.name != "initial_dataset":
            error_message = f"No such subset of initial_dataset at {subset_path}!"
            logger.error(error_message)
        else:
            info_message = f"Subset at {subset_path} found!"
            logger.info(info_message)

    if False in existent_subset_paths:
        return 1

    if final_dataset_path.exists():
        info_message = f"Final dataset folder at {final_dataset_path} already exists!"
        logger.info(info_message)
        return 1

    final_dataset_path.mkdir()

    return 0


if __name__ == "__main__":
    sys.exit(divide_initial_dataset())
