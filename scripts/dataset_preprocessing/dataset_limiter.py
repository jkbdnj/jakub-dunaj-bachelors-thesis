"""Script limiting the number of images in each class to min(true_number, limit).

This limiting happens in the unprocessed original dataset from Mohanty. This script crates the
initial dataset that will be used for further preprocessing and division into train and validate
datasets.

"""

import logging
import random
import sys
from pathlib import Path
from typing import Literal

from PIL import Image

file_name = Path(__file__).name

if __name__ != "__main__":
    logger = logging.getLogger("wrapper." + __name__)
else:
    logger = logging.getLogger(__name__)
    FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename="dataset_limiter.log", format=FORMAT, level=logging.INFO)

# used for selection of images in the original dataset
LIMIT = 200

# constant for the path to the original color subset
ORIGINAL_COLOR_PATH = Path("PATH_TO/datasets/original_dataset/color")

# constant for the path to the original color subset
ORIGINAL_SEGMENTED_PATH = Path("PATH_TO/datasets/original_dataset/segmented")

# constant for the path to the initial dataset (containing selected images from original dataset)
INITIAL_PATH = Path("PATH_TO/datasets/initial_dataset/segmented")


def limit_dataset(
    color_path: Path, segmented_path: Path, destination_path: Path, limit: int
) -> Literal[0, 1]:
    """Function limiting the number of images in each class of subsets given as arguments.

    Args:
        color_path (Path): The path to the color subset of the original dataset.
        segmented_path (Path): The path to the color subset of the original dataset.
        destination_path (Path): The path where the subsets given will be saved.
        limit (int): Used to limit the number of images in a class to min(true_number, limit).

    Returns:
        Literal[0, 1]: Returns 0 on success, 1 on failure.

    """
    info_message = f"Running limit_dataset(...) function at {file_name}"
    logger.info(info_message)
    subset_paths = [color_path, segmented_path]
    subset_paths_exist = [subset_path.exists() for subset_path in subset_paths]

    for subset_path, exists in zip(subset_paths, subset_paths_exist, strict=False):
        if not exists:
            error_message = f"No such folder at path {subset_path} found!"
            logger.error(error_message)
        else:
            info_message = f"Folder at path {subset_path} found!"
            logger.info(info_message)

    if False in subset_paths_exist:
        return 1

    if destination_path.exists():
        error_message = f"Initial dataset folder at path {destination_path} already exists!"
        logger.error(error_message)
        return 1

    destination_path.mkdir()

    for subset_path in subset_paths:
        destination_subset_path = destination_path / subset_path.name
        destination_subset_path.mkdir()
        for class_path in subset_path.iterdir():
            if class_path.is_dir():
                info_message = (
                    f"Selecting images in class {class_path.name} at {subset_path.name} dataset."
                )
                logger.info(info_message)

                destination_class_path = destination_subset_path / class_path.name
                destination_class_path.mkdir()

                image_paths = list(class_path.glob("*.[jJ][pP][gG]"))

                # randomly reordering the paths prior to the selection
                random.shuffle(image_paths)

                image_paths = image_paths if len(image_paths) <= limit else image_paths[:limit]
                for image_path in image_paths:
                    with Image.open(image_path) as image:
                        image.save(destination_class_path / image_path.name)

            info_message = (
                f"Image selection in class {class_path.name} at {subset_path.name} dataset done."
            )
            logger.info(info_message)

    logger.info("Dataset selection successful!")
    return 0


if __name__ == "__main__":
    info_message = f"Running {file_name} directly as script."
    logger.info(info_message)
    sys.exit(limit_dataset(ORIGINAL_COLOR_PATH, ORIGINAL_SEGMENTED_PATH, INITIAL_PATH, LIMIT))
