"""File holding simple command line utilities."""

from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

from rich_argparse import RichHelpFormatter


def validate_dataset_path(arg_value: str, arg: str) -> Path:
    """Function validating the path to a dataset (train/test) folder."""
    path = Path(arg_value)
    if not path.exists():
        error_message = f"Path {path} for argument {arg} does not exist!"
        raise ArgumentTypeError(error_message)

    return path


def validate_output_path(arg: str) -> Path:
    """Function validating output path."""
    path = Path(arg)
    if not path.exists():
        path.mkdir()

    return path


def validate_int(arg_value: str, arg: str) -> int:
    """Function validating integers."""
    try:
        value = int(arg_value)

        if value <= 0:
            error_message = (
                f'Value "{arg_value}" of the argument {arg} is not valid! '
                "It must be a positive integer."
            )
            raise ArgumentTypeError(error_message)

        return value

    except ValueError:
        error_message = f'Value "{arg_value}" of the argument {arg} is not an integer!'
        raise ArgumentTypeError(error_message) from None


def parse_args():
    """Function parsing command line arguments."""
    parser = ArgumentParser(
        prog="model_trainer",
        description="""This program trains an EfficientNetB0 CNN model
        to classify diseased :seedling: images.""",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "train_dataset_path",
        help="path to the train dataset",
        type=lambda arg_value: validate_dataset_path(arg_value, "train_dataset_path"),
    )
    parser.add_argument(
        "test_dataset_path",
        help="path to the test dataset",
        type=lambda arg_value: validate_dataset_path(arg_value, "test_dataset_path"),
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="batch size",
        type=lambda arg_value: validate_int(arg_value, arg="-b/--batch_size"),
        default=32,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="number of epochs",
        type=lambda arg_value: validate_int(arg_value, arg="-e/--epochs"),
        default=10,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output folder of the metrics, if it does not exist, it will be created",
        required=False,
        default=Path("./results"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
