"""File holding simple command line utilities."""

import logging
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

from rich_argparse import RichHelpFormatter

logger = logging.getLogger("root." + __name__)


def validate_dataset_path(arg_value: str, arg: str) -> Path:
    """Function validating the path to a dataset (train/test) folder.

    This function checks the existence of the train/test datasets.

    Args:
        arg (str): Argument as a string.
        arg_value (str): The value of the argument.

    Returns:
        Path: Returns a Path object representing the path to the train/test dataset.

    """
    path = Path(arg_value)
    if not path.exists():
        error_message = f"Path {path} for argument {arg} does not exist!"
        logging.error(error_message)
        raise ArgumentTypeError(error_message)

    return path


def validate_int(arg_value: str, arg: str) -> int:
    """Function validating integers.

    This function validates integer values. The values should be positive integers.
    If the value is non-integer value, negative integer or 0, it raises an
    ArgumentTypeError exception.

    Args:
        arg (str): Argument as a string.
        arg_value (str): The value of the argument.

    Returns:
        int: Returns the integer value of the argument value.

    """
    try:
        value = int(arg_value)

        if value <= 0:
            error_message = (
                f'Value "{arg_value}" of the argument {arg} is not valid! '
                "It must be a positive integer."
            )
            logger.error(error_message)
            raise ArgumentTypeError(error_message)

        return value

    except ValueError:
        error_message = f'Value "{arg_value}" of the argument {arg} is not an integer!'
        logger.error(error_message)
        raise ArgumentTypeError(error_message) from None


def validate_output_path(output_path: Path | None) -> Path:
    """Function validating output path.

    This function validates the output path. If an output path is given, it checks its
    existence and appends a /results folder to it. If an output is not given, it crates
    a folder /results in current working directory.

    Args:
        output_path (Path|None): The output path.

    Returns:
        Path: Returns a Path object representing the path to the results folder.

    """
    if output_path is None:
        # creates a default path
        output_path = Path("./results")
        output_path.mkdir(exist_ok=True)
        return output_path

    # creates folder results at the output path if it does not exists
    output_path = output_path / "results"

    try:
        output_path.mkdir(exist_ok=True)
    except FileNotFoundError:
        error_message = f"Folder at path {output_path} does not exist!"
        logger.error(error_message)
        raise ArgumentTypeError(error_message) from None
    return output_path


def parse_args():
    """Function parsing command line arguments.

    This function crates the parser object and adds positional and optional arguments to the parser.
    Next it parsers the command line input.

    Returns:
        args (Namespace): Object holding parsed command line arguments.

    """
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
        type=Path,
        default=None,
    )

    args = parser.parse_args()
    args.output = validate_output_path(args.output)

    return args
