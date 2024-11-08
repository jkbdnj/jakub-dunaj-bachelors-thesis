"""Script dividing the initial dataset into training and testing subsets."""

import logging
import random
import sys
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="divide_initial_dataset.log", format=FORMAT, level=logging.INFO)

# variable defining the train-test ratio of the initial dataset
TRAIN_TEST_RATIO = 0.8


def divide_initial_dataset(train_test_ratio: float):
    """This function divides the initial dataset into training and testing subsets."""
    color_path = Path(
        "/Users/kubkodunaj/Desktop/jakub-dunaj-bachelors-thesis/datasets/initial_dataset/color"
    )
    segmented_path = Path(
        "/Users/kubkodunaj/Desktop/jakub-dunaj-bachelors-thesis/datasets/initial_dataset/segmented"
    )
    artificial_path = Path(
        "/Users/kubkodunaj/Desktop/jakub-dunaj-bachelors-thesis/datasets/initial_dataset/artificial_background"
    )
    final_dataset_path = Path(
        "/Users/kubkodunaj/Desktop/jakub-dunaj-bachelors-thesis/datasets/final_dataset"
    )

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
        error_message = f"Final dataset folder at {final_dataset_path} already exists!"
        logger.info(error_message)
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
        segmented_class_path = segmented_path / image_class
        color_class_path = color_path / image_class
        artificial_class_path = artificial_path / image_class
        class_paths = [segmented_class_path, color_class_path, artificial_class_path]
        class_paths_exist = [class_path.exists() for class_path in class_paths]
        for class_path, exists in zip(class_paths, class_paths_exist, strict=False):
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

        for class_path in class_paths:
            image_paths = list(class_path.glob("*.[jJ][pP][gG]"))
            random.shuffle(image_paths)
            division_index = int(len(image_paths) * train_test_ratio)

            for image_path in image_paths[:division_index]:
                with Image.open(image_path) as image:
                    image.save(final_train_path / class_path.name / image_path.name)

            for image_path in image_paths[division_index:]:
                with Image.open(image_path) as image:
                    image.save(final_test_path / class_path.name / image_path.name)

    logger.info("Dataset division process successful!")
    return 0


if __name__ == "__main__":
    sys.exit(divide_initial_dataset(TRAIN_TEST_RATIO))
