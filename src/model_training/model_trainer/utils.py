"""Module providing utility function for the model_trainer package."""

import time
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy
from keras.src import ops
from matplotlib import ticker


def generate_time_stamp():
    """Function generating the time stamp following the ISO 8601 format."""
    time_stamp = time.time()
    return time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime(time_stamp))


def save_bar_plot(
    labels: list[str],
    data: list[float],
    title: str,
    file_name: str,
    file_format: str,
    output_path: Path,
) -> None:
    """Function saving the data and labels as bar plot.

    The function generates a figure with plotted and labeled data. It also adds a time stamp
    to the file name.

    Args:
        labels (list[str]): Labels for the bars in the bar plot.
        data (list[float]): Data to be plotted.
        title (str): Title of the bar plot.
        file_name (str): File name of the figure to be saved.
        file_format (str): Format of the figure to be saved.
        output_path (Path): The destination where the figure will be saved.

    """
    figure, ax = plt.subplots(figsize=(20, 10))
    plt.subplots_adjust(left=0.2, right=0.9, top=0.96, bottom=0.04)

    # setting up x-axis label
    ax.set(xlabel="accuracy")

    # averaging the data
    average = numpy.mean(data)

    # adding the respective values to the bars
    bars = ax.barh(labels, data)
    ax.bar_label(bars, fmt="%.2f")

    # inverting the y-axis so that the labels start from the top of the bar plot
    ax.invert_yaxis()

    # plotting the average as a line
    ax.axvline(x=average, color="g", label="average", lw=2)

    # adding legend for the average line
    ax.legend(loc="best")

    # adding tick for the average line to the x-axis
    ticks = list(ax.get_xticks()) + [average]
    ticks.sort()

    # formatting the ticks at the x-axis
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    ax.set_xticks(ticks)

    # setting title
    ax.set_title(title, fontsize="x-large")

    # formatting file name
    time_stamp = generate_time_stamp()
    file_name = f"{file_name}_{time_stamp}.{file_format}"

    # saving and closing the figure
    plt.tight_layout()
    figure.savefig(output_path / file_name, format=file_format)
    plt.close(figure)


def save_accuracy_per_class_as_plot(
    history: keras.callbacks.History, labels: list, output_path: Path
) -> None:
    """Nothing now."""
    train_accuracy_matrix = []

    # iterating over tf.Tensor (not symbolic)
    for train_accuracy_epoch in history.history["accuracy_per_class"]:
        train_accuracy_matrix.append(ops.convert_to_numpy(train_accuracy_epoch))

    train_mean_per_class = numpy.mean(train_accuracy_matrix, axis=0)

    # saving the bar plot with train accuracy per class averaged over all epochs
    save_bar_plot(
        labels=labels,
        data=train_mean_per_class,
        title="Train accuracy per class averaged over all epochs",
        file_name="train_accuracy_per_class",
        file_format="png",
        output_path=output_path,
    )

    test_accuracy_matrix = []

    # iterating over tf.Tensor (not symbolic)
    for test_accuracy_epoch in history.history["val_accuracy_per_class"]:
        test_accuracy_matrix.append(ops.convert_to_numpy(test_accuracy_epoch))

    val_mean_per_class = numpy.mean(test_accuracy_matrix, axis=0)

    # saving the bar plot with validation accuracy per class averaged over all epochs
    save_bar_plot(
        labels=labels,
        data=val_mean_per_class,
        title="Test accuracy per class averaged over all epochs",
        file_name="test_accuracy_per_class",
        file_format="png",
        output_path=output_path,
    )


def save_accuracy_and_loss_as_plot(history: keras.callbacks.History, output_path: Path) -> None:
    """Function saving accuracy and loss metrics as plots.

    This function aggregates the train and test accuracy and train and test loss into two
    separate subplots of a figure

    Args:
        history (keras.callbacks.History): Object holding accuracy and loss metrics form the
        training process.
        output_path (Path): Output file path, where the plot is saved.

    """
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.2, right=0.9, top=0.96, bottom=0.04)
    epochs = range(1, len(history.history["accuracy"]) + 1)

    # subplot for training and testing accuracy
    ax1.set(xlabel="epoch", ylabel="accuracy")
    ax1.set_title("Model Accuracy")
    ax1.plot(epochs, history.history["accuracy"], label="train accuracy")
    ax1.plot(epochs, history.history["val_accuracy"], label="test accuracy")
    ax1.set_xticks(epochs)
    ax1.legend()

    # subplot for training and testing loss
    ax2.set(xlabel="epoch", ylabel="loss")
    ax2.set_title("Model Loss")
    ax2.plot(epochs, history.history["loss"], label="train loss")
    ax2.plot(epochs, history.history["val_loss"], label="test loss")
    ax2.set_xticks(epochs)
    ax2.legend()

    # formatting file name
    time_stamp = generate_time_stamp()
    file_name = f"accuracy_and_loss_{time_stamp}.png"

    plt.tight_layout()
    figure.savefig(output_path / file_name, format="png")
    plt.close(figure)


def save_validation_metrics_as_csv(output_path: Path, metrics: dir) -> None:
    """Function saving the validation metrics as csv file.

    Args:
        output_path (Path): Output file path, where the csv is saved.
        metrics (dir): Validation metrics as directory.

    """
    # with output_path.open("w") as file:
    #     fieldnames = ["metric", "value"]
    #     writer = csv.DictWriter(file, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for k, v in metrics.items():
    #         writer.writerow({"metric": k, "value": v})


def save_history_as_json(history: keras.callbacks.History, output_path: Path) -> None:
    """Nothing now."""
