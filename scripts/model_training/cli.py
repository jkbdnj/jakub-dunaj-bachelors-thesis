"""Simple class holding command line utilities."""

from argparse import (
    ArgumentParser,
    ArgumentTypeError,
)
from pathlib import Path

from rich_argparse import RichHelpFormatter


def validate_dataset_path(arg: str) -> Path:
    """Function validating the path to a dataset (train/test) folder."""
    path = Path(arg)
    if not path.exists():
        error_message = f"Path {path} does not exist!"
        raise ArgumentTypeError(error_message)

    return path


def validate_output_path(arg: str) -> Path:
    """Function validating output path."""
    path = Path(arg)
    if not path.exists():
        path.mkdir()

    return path


def validate_epochs(arg: str) -> int:
    """Function validating epochs."""
    try:
        epochs = int(arg)

        if epochs <= 0:
            error_message = (
                f"Value {arg} of the option -e/--epochs is not valid! "
                "It must be a positive integer."
            )
            raise ArgumentTypeError(error_message)

        return epochs

    except ValueError:
        error_message = f"Value {arg} of the option -e/--epochs is not an integer!"
        raise ArgumentTypeError(error_message) from None


def parse():
    """Function parsing command line arguments."""
    parser = ArgumentParser(
        prog="model_trainer",
        description="""This program trains an EfficientNetB0 CNN model
        to classify :seedling: images containing .""",
        formatter_class=RichHelpFormatter,
    )

    parser.add_argument(
        "train_folder",
        help="path to the train dataset",
        type=validate_dataset_path,
    )
    parser.add_argument(
        "test_folder",
        help="path to the test dataset",
        type=validate_dataset_path,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="number of epochs",
        type=validate_epochs,
        default=10,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output folder of the metrics, if it does not exist, it will be created",
        required=False,
        default="./results",
    )

    return parser.parse_args()


if __name__ == "__main__":
    parse()
