"""Script as a wrapper for artificial_background_adder.py and initial_dataset_divider.py modules."""

import logging
import sys
from pathlib import Path

from artificial_background_adder import add_artificial_background
from initial_dataset_divider import divide_initial_dataset

logger = logging.getLogger("wrapper")
FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="dataset_preprocessing.log", format=FORMAT, level=logging.INFO)

# constant for the path to the folder with artificial background images
ARTIFICIAL_BACKGROUNDS_PATH = Path("PATH_TO/scripts/dataset_preprocessing/artificial_backgrounds")

# constant for the path to the segmented subset of the initial dataset
SEGMENTED_PATH = Path("PATH_TO/datasets/initial_dataset/segmented")

# constant for the path where the augmented images are saved
DESTINATION_PATH = SEGMENTED_PATH.parent / "artificial_background"

# constant defining the train-test ratio of the initial dataset if the module is called as a script
TRAIN_TEST_RATIO = 0.8

# constant for the path to the color subset of the initial dataset
COLOR_PATH = Path("PATH_TO/datasets/initial_dataset/color")

# constant for the path to the artificial background subset of the initial dataset
ARTIFICIAL_PATH = Path("PATH_TO/datasets/initial_dataset/artificial_background")

# constant for the path where the final dataset used for the training process is saved
FINAL_DATASET_PATH = Path("PATH_TO/datasets/final_dataset")


def main():
    """Main function.

    This function executes the augmentation of segmented images with artificial background images
    and the division of the dataset into training and testing subsets.

    """
    logger.info("Running dataset augmentation.")
    result_augmentation = add_artificial_background(
        ARTIFICIAL_BACKGROUNDS_PATH, SEGMENTED_PATH, DESTINATION_PATH
    )
    info_message = f"Dataset augmentation done with return code {result_augmentation}."
    logger.info(info_message)
    if not result_augmentation:
        logger.info("Running dataset division.")
        result_division = divide_initial_dataset(
            TRAIN_TEST_RATIO, COLOR_PATH, SEGMENTED_PATH, ARTIFICIAL_PATH, FINAL_DATASET_PATH
        )
        info_message = f"Dataset division done with return code {result_division}."
        logger.info(info_message)

        return result_division
    return result_augmentation


if __name__ == "__main__":
    sys.exit(main())
