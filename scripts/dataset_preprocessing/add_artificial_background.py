"""Script adding artificial backgrounds to segmented leaf images.

This script strictly adheres to the layout of the initial dataset. It takes segmented leaf images
from the initial dataset and creates a new subset of images with added artificial backgrounds.
"""

import logging
import sys
from pathlib import Path

import numpy
from PIL import Image

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="add_artificial_background.log", format=FORMAT, level=logging.INFO)


def add_artificial_background():
    """This function adds artificial backgrounds to segmented leaf images.

    The function loads images form the segmented subset of the initial dataset and adds
    artificial background from the folder holding artificial background images. The
    folder structure of the segmented subset is copied and the images are saved to a
    defined destination.
    """
    artificial_backgrounds_folder = Path("ABSOLUTE_PATH_TO/cnn_model/artificial_backgrounds")
    segmented_images_folder = Path("ABSOLUTE_PATH_TO/initial_dataset/segmented")
    destination_folder = Path("ABSOLUTE_PATH_TO/initial_dataset/artificial_background")

    if not artificial_backgrounds_folder.exists():
        error_message = f"No such folder with background images:{artificial_backgrounds_folder}"
        logger.error(error_message)
        return 1

    if not segmented_images_folder.exists():
        error_message = f"No such folder with segmented images:{segmented_images_folder}"
        logger.error(error_message)
        return 1

    if destination_folder.exists():
        error_message = f"Destination folder at {destination_folder} already exists!"
        logger.info(error_message)
        return 1

    destination_folder.mkdir()

    background_images = []
    for image_path in artificial_backgrounds_folder.glob("*.jpg"):
        with Image.open(image_path) as background_image:
            # resizing the background image to 256 * 256 pixels using the bilinear interpolation
            resized_background_image = background_image.resize((256, 256), Image.BILINEAR)
            background_images.append(numpy.array(resized_background_image))

    for class_folder in segmented_images_folder.iterdir():
        if class_folder.is_dir():
            info_message = f"Augmenting images in class {class_folder}."
            logger.info(info_message)

            # creating a folder with the same name in the destination
            (destination_folder / class_folder.name).mkdir()

            # counter to cycle through background images
            i = 0

            for image_path in class_folder.glob("*.jpg"):
                with Image.open(image_path) as image:
                    # numpy.array() creates a copy of the image passed
                    # using numpy.asarray() would create only a read-only view
                    image_as_array = numpy.array(image)

                    # extracting color channels and crating boolean mask with information where
                    # black pixels are in the image
                    threshold_value = 15
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
                    augmented_image.save(
                        destination_folder / class_folder.name / augmented_image_name
                    )

                i = (i + 1) % len(background_images)

            info_message = f"Done adding artificial background images in class {class_folder}."
            logger.info(info_message)

    logger.info("Augmentation process successful!")
    return 0


if __name__ == "__main__":
    sys.exit(add_artificial_background())
