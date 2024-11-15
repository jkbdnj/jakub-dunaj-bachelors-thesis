"""Script dividing the initial dataset into training and testing subsets.

This script strictly adheres to the structure of the initial dataset. Based on the train-test ratio
that is set up in the script, the script divides every class in each of the three dataset subsets
based on this ratio and mixes the images of the same calls from the three different subsets into one
single class in the final dataset.

"""

import logging
import random
import sys
from pathlib import Path

from PIL import Image

if __name__ != "__main__":
    logger = logging.getLogger("wrapper." + __name__)
else:
    logger = logging.getLogger(__name__)
    FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename="initial_dataset_divider.log", format=FORMAT, level=logging.INFO)


# constant defining the train-test ratio of the initial dataset if the module is called as a script
TRAIN_TEST_RATIO = 0.8

# constant for the path to the color subset of the initial dataset
COLOR_PATH = Path("PATH_TO/initial_dataset/color")

# constant for the path to the segmented subset of the initial dataset
SEGMENTED_PATH = Path("PATH_TO/initial_dataset/segmented")

# constant for the path to the artificial background subset of the initial dataset
ARTIFICIAL_PATH = Path("PATH_TO/initial_dataset/artificial_background")

# constant for the path where the final dataset used for the training process is saved
FINAL_DATASET_PATH = Path("PATH_TO/final_dataset")


def divide_initial_dataset(
    train_test_ratio: float,
    color_path: Path,
    segmented_path: Path,
    artificial_path: Path,
    final_dataset_path: Path,
):
    """Function dividing the initial dataset into training and testing subsets.

    It takes images from each class of the three datasets and divides them into training and testing
    subsets based on teh train-test ration provided as function parameter.

    Args:
        train_test_ratio (float): The parameter expressing what part of the dataset will be used for
        training and what part for testing.
        color_path (Path): The path to the color subset of the initial dataset.
        segmented_path (Path): The path to the segmented subset of the initial dataset.
        artificial_path (Path): The path to the artificial background subset of the initial dataset.
        final_dataset_path (Path): The destination path for the newly created dataset divided into
        training and testing subsets.

    """
    subset_paths = [color_path, segmented_path, artificial_path]
    subset_paths_exist = [folder.exists() for folder in subset_paths]

    for subset_path, exists in zip(subset_paths, subset_paths_exist, strict=False):
        if not exists or subset_path.parent.name != "initial_dataset":
            error_message = f"No such subset of initial_dataset at {subset_path} found!"
            logger.error(error_message)
        else:
            info_message = f"Subset at {subset_path} found!"
            logger.info(info_message)

    if False in subset_paths_exist:
        return 1

    if final_dataset_path.exists():
        error_message = f"Final dataset folder at path {final_dataset_path} already exists!"
        logger.error(error_message)
        return 1

    final_dataset_path.mkdir()
    final_train_path = final_dataset_path / "train"
    final_test_path = final_dataset_path / "test"
    final_train_path.mkdir()
    final_test_path.mkdir()

    image_classes = [class_path.name for class_path in color_path.iterdir() if class_path.is_dir()]

    for image_class in image_classes:
        (final_train_path / image_class).mkdir()
        (final_test_path / image_class).mkdir()

    for image_class in image_classes:
        # crating the paths a class in all three dataset subsets
        segmented_class_path = segmented_path / image_class
        color_class_path = color_path / image_class
        artificial_class_path = artificial_path / image_class

        # verifying if the class is present in all three subsets
        class_paths = [
            segmented_class_path,
            color_class_path,
            artificial_class_path,
        ]
        class_paths_exist = [class_path.exists() for class_path in class_paths]
        for class_path, exists in zip(
            class_paths,
            class_paths_exist,
            strict=False,
        ):
            if not exists:
                error_message = (
                    f"No class {class_path.name} of {class_path.parent.name} subset found!"
                )
                logger.error(error_message)
            else:
                info_message = f"Class {class_path.name} of {class_path.parent.name} subset found!"
                logger.info(info_message)

        if False in class_paths_exist:
            return 1

        # dividing images of a class from all three datasets
        for class_path in class_paths:
            info_message = (
                f"Dividing images in class {class_path.name} of {class_path.parent.name} subset."
            )
            logger.info(info_message)

            image_paths = list(class_path.glob("*.[jJ][pP][gG]"))

            # randomly reordering the paths prior to the division
            random.shuffle(image_paths)

            # division index if the list with paths
            division_index = int(len(image_paths) * train_test_ratio)

            # training subset
            for image_path in image_paths[:division_index]:
                with Image.open(image_path) as image:
                    image.save(final_train_path / class_path.name / image_path.name)

            # testing subset
            for image_path in image_paths[division_index:]:
                with Image.open(image_path) as image:
                    image.save(final_test_path / class_path.name / image_path.name)

            info_message = (
                f"Done dividing images in class {class_path.name} of "
                f"{class_path.parent.name} subset."
            )
            logger.info(info_message)

    logger.info("Dataset division process into training and testing subsets successful!")
    return 0


if __name__ == "__main__":
    sys.exit(
        divide_initial_dataset(
            TRAIN_TEST_RATIO, COLOR_PATH, SEGMENTED_PATH, ARTIFICIAL_PATH, FINAL_DATASET_PATH
        )
    )
