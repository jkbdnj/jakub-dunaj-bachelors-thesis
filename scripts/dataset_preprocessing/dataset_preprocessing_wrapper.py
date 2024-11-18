"""Script preprocessing the original dataset."""

import logging
import sys
from pathlib import Path
from typing import Literal

from artificial_background_adder import add_artificial_background
from dataset_divider import divide_dataset
from dataset_limiter import limit_dataset

logger = logging.getLogger("wrapper")
FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="dataset_preprocessing.log", format=FORMAT, level=logging.INFO)

# used for selection of images in the original dataset
LIMIT = 200

# constant for the path to the original color subset
ORIGINAL_COLOR_PATH = Path("PATH_TO/datasets/original_dataset/color")

# constant for the path to the original color subset
ORIGINAL_SEGMENTED_PATH = Path("PATH_TO/datasets/original_dataset/segmented")

# constant for the path to the initial dataset (containing selected images from original dataset)
INITIAL_PATH = Path("PATH_TO/datasets/initial_dataset")

# constant for the path to the folder with artificial background images
ARTIFICIAL_BACKGROUNDS_PATH = Path("PATH_TO/scripts/dataset_preprocessing/artificial_backgrounds")

# constant for the path to the segmented subset of the initial dataset
SEGMENTED_PATH = Path("PATH_TO/datasets/initial_dataset/segmented")

# constant for the path to the artificial background subset of the initial dataset
ARTIFICIAL_PATH = Path("PATH_TO/datasets/initial_dataset/artificial_background")

# constant for the path to the color subset of the initial dataset
COLOR_PATH = Path("PATH_TO/datasets/initial_dataset/color")

# constant defining the train-validation ratio of the initial dataset
TRAIN_VALIDATION_RATIO = 0.8

# constant for the path where the final dataset used for the training process is saved
FINAL_DATASET_PATH = Path("PATH_TO/datasets/final_dataset")


def main() -> Literal[0, 1]:
    """Main function.

    This function executes the selection of images of the original dataset, augmentation of
    segmented images with artificial background images and the division of the dataset
    into training and testing subsets.

    The process: original dataset (selection of images) -> initial dataset with augmented images
    -> final dataset (splitted into train and validation datasets)

    Returns:
        Literal[0, 1]: Returns 0 on success, 1 on failure.

    """
    pipeline = [
        (limit_dataset, (ORIGINAL_COLOR_PATH, ORIGINAL_SEGMENTED_PATH, INITIAL_PATH, LIMIT)),
        (
            add_artificial_background,
            (ARTIFICIAL_BACKGROUNDS_PATH, SEGMENTED_PATH, ARTIFICIAL_PATH),
        ),
        (
            divide_dataset,
            (
                TRAIN_VALIDATION_RATIO,
                COLOR_PATH,
                SEGMENTED_PATH,
                ARTIFICIAL_PATH,
                FINAL_DATASET_PATH,
            ),
        ),
    ]

    for func, args in pipeline:
        info_message = f"Executing function {func.__name__}."
        logger.info(info_message)

        # executing the function with unpacked arguments
        result = func(*args)
        if result == 1:
            error_message = f"Execution of the function {func.__name__} failed!"
            logger.error(error_message)

            return 1

        info_message = f"Execution of the function {func.__name__} successful."
        logger.info(info_message)

    return 0


if __name__ == "__main__":
    sys.exit(main())
