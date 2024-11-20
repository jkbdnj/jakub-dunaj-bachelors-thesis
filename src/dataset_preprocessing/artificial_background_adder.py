"""Script adding artificial backgrounds to segmented leaf images.

This script strictly adheres to the layout of the initial dataset. It takes segmented leaf images
from the initial dataset and creates a new subset of images with added artificial backgrounds.

"""

import logging
import sys
from pathlib import Path
from typing import Literal

import numpy
from PIL import Image

file_name = Path(__file__).name

if __name__ != "__main__":
    logger = logging.getLogger("wrapper." + __name__)
else:
    logger = logging.getLogger(__name__)
    FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        filename="artificial_background_adder.log", format=FORMAT, level=logging.INFO
    )

# constant for the path to the folder with artificial background images
ARTIFICIAL_BACKGROUNDS_PATH = Path("PATH_TO/scripts/dataset_preprocessing/artificial_backgrounds")

# constant for the path to the segmented subset of the initial dataset
SEGMENTED_PATH = Path("PATH_TO/initial_dataset/segmented")

# constant for the path where the augmented images are saved
DESTINATION_PATH = Path(SEGMENTED_PATH.parent / "artificial_background")


def add_artificial_background(
    artificial_backgrounds_path: Path, segmented_path: Path, destination_path: Path
) -> Literal[0, 1]:
    """Function adding artificial backgrounds to segmented leaf images.

    The function loads images form the segmented subset of the initial dataset and adds artificial
    background from the folder holding artificial background images. The folder structure of the
    segmented subset is copied and the images with added artificial backgrounds are saved to a
    defined destination.

    Args:
        artificial_backgrounds_path (Path): The path to the folder with artificial
        background images.
        segmented_path (Path): The path to the segmented subset of the initial dataset.
        destination_path (Path): The path where the augmented images are saved.

    Returns:
        Literal[0, 1]: Returns 0 on success, 1 on failure.

    """
    info_message = f"Running add_artificial_background(...) function at {file_name}"
    logger.info(info_message)
    paths = [artificial_backgrounds_path, segmented_path]
    paths_exist = [path.exists() for path in paths]

    for path, exists in zip(paths, paths_exist, strict=False):
        if not exists:
            error_message = f"No such folder at path {path} found!"
            logger.error(error_message)
        else:
            info_message = f"Folder at path {path} found!"
            logger.info(info_message)

    if False in paths_exist:
        return 1

    if destination_path.exists():
        error_message = f"Destination folder at path {destination_path} already exists!"
        logger.error(error_message)
        return 1

    destination_path.mkdir()

    background_images = []
    for image_path in artificial_backgrounds_path.glob("*.jpg"):
        with Image.open(image_path) as background_image:
            # resizing the background image to 256 * 256 pixels using the bilinear interpolation
            resized_background_image = background_image.resize(
                (256, 256), Image.Resampling.BILINEAR
            )
            background_images.append(numpy.array(resized_background_image))

    for class_path in segmented_path.iterdir():
        if class_path.is_dir():
            info_message = f"Augmenting images in class {class_path.name}."
            logger.info(info_message)

            # creating a folder with the same name in the destination
            (destination_path / class_path.name).mkdir()

            # counter to cycle through background images
            i = 0

            for image_path in class_path.glob("*.jpg"):
                with Image.open(image_path) as image:
                    # numpy.array() creates a copy of the image passed
                    # using numpy.asarray() would create only a read-only view
                    image_as_array = numpy.array(image)

                    # this value enables for smooth pixel change around leaf outlines
                    threshold_value = 16

                    # extracting color channels and crating boolean mask with information where
                    # black pixels are in the image
                    red_channel = image_as_array[:, :, 0]
                    green_channel = image_as_array[:, :, 1]
                    blue_channel = image_as_array[:, :, 2]
                    black_pixels = (
                        (red_channel[:, :] < threshold_value)
                        & (green_channel[:, :] < threshold_value)
                        & (blue_channel[:, :] < threshold_value)
                    )

                    # pixels that are black are replaced with pixels from the background image
                    image_as_array[black_pixels] = background_images[i][black_pixels]

                    # converting the array to Image object and saving the augmented image
                    augmented_image = Image.fromarray(image_as_array)
                    augmented_image_name = image_path.name.replace(
                        "final_masked", "artificial_background"
                    )
                    augmented_image.save(destination_path / class_path.name / augmented_image_name)

                i = (i + 1) % len(background_images)

            info_message = (
                f"Done adding artificial background images to images in class {class_path.name}."
            )
            logger.info(info_message)

    logger.info("Augmentation process successful!")
    return 0


if __name__ == "__main__":
    info_message = f"Running {file_name} directly as script."
    logger.info(info_message)
    sys.exit(
        add_artificial_background(ARTIFICIAL_BACKGROUNDS_PATH, SEGMENTED_PATH, DESTINATION_PATH)
    )
