"""Module providing training functionalities to train EfficientNetB0 CNN model.

This module provides training functionalities to train EfficientNetB0 CNN model for plant disease
classification from leaf images.

"""

import csv
import logging
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import EfficientNetB0
from keras.callbacks import CSVLogger
from keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Pipeline,
    RandomBrightness,
    RandomContrast,
    RandomFlip,
    RandomRotation,
)
from keras.utils import image_dataset_from_directory

from .cli import CLI
from .exceptions import DatasetError

logger = logging.getLogger(__name__)


class ModelTrainer:
    """This class represents provides model training functionality.

    Attributes:
        train_dataset (tf.data.Dataset): The dataset used for training.
        test_dataset (tf.data.Dataset): The dataset used for testing.
        validation_dataset (tf.data.Dataset): The dataset used for validation.
        batch_size (int): The batch size used.

    """

    def __init__(self, train_dataset_path: Path, validation_dataset_path: Path, batch_size: int):
        """Constructor instantiating an object of the ModelTrainer class.

        This constructor loads up the train, test and validation datasets and applies augmentation
        pipeline to each of the datasets. It also sets up the instance variables train_dataset,
        test_dataset and validation_dataset.

        Args:
            train_dataset_path (Path): The path to the train dataset.
            validation_dataset_path (Path): The path to the validation dataset.
            batch_size (int): The batch size used.

        """
        train_dataset, test_dataset, validation_dataset = self.__load_dataset(
            train_dataset_path, validation_dataset_path, batch_size
        )
        self.train_dataset = self.__apply_augmenting_pipeline(train_dataset)
        self.test_dataset = self.__apply_augmenting_pipeline(test_dataset)
        self.validation_dataset = self.__apply_augmenting_pipeline(validation_dataset)
        self.batch_size = batch_size

    def __load_dataset(
        self, train_dataset_path: Path, validation_dataset_path: Path, batch_size: int
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Private method loading train, test and validation datasets.

        Args:
            train_dataset_path (Path): The path to the train dataset.
            validation_dataset_path (Path):  The path to the test dataset.
            batch_size (int): The batch size.

        Returns:
            tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Tuple with train, test and
            validation datasets.

        Raises:
            DatasetError: If the classes in train, test and validation datasets differ from each
            other.

        """
        train_dataset, test_dataset = image_dataset_from_directory(
            train_dataset_path,
            labels="inferred",
            label_mode="int",
            color_mode="rgb",
            batch_size=batch_size,
            seed=1234,
            validation_split=0.2,
            subset="both",
            image_size=(224, 224),
            interpolation="bilinear",
        )

        validation_dataset = image_dataset_from_directory(
            validation_dataset_path,
            labels="inferred",
            label_mode="int",
            color_mode="rgb",
            batch_size=batch_size,
            image_size=(224, 224),
            interpolation="bilinear",
        )

        if not (
            set(train_dataset.class_names)
            == set(test_dataset.class_names)
            == set(validation_dataset.class_names)
        ):
            error_message = "Train, test, and validation datasets differ in the class names!"
            raise DatasetError(error_message)

        return train_dataset, test_dataset, validation_dataset

    def __apply_augmenting_pipeline(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Private method applying augmentation operations to dataset images.

        This method uses an augmentation pipeline with 4 augmentation layers. These layers are:
        RandomFlip, RandomRotation, RandomContrast, and RandomBrightness. Each layer applies an
        augmentation operation. The randomness lies in the range of possible operation
        configurations (e.g. the range of brightness increase or decrease).

        Setting the fill_mode as constant inn RandomRotation will fill the missing information
        with 0s. This adds the least artificial information that might otherwise add unwanted
        distortion to the dataset. Setting the factor to 0.2 in RandomContrast will change the
        contrast of a pixel only slightly. The same goes for RandomBrightness.

        Args:
            dataset (tf.data.Dataset): Dataset that will be augmented using the augmentation
            pipeline.

        Returns:
            tf.data.Dataset: Augmented dataset by the defined augmentation pipeline.

        """
        augmentation_pipeline = Pipeline(
            [
                RandomFlip(mode="horizontal_and_vertical"),
                RandomRotation(factor=0.1, fill_mode="constant"),
                RandomContrast(factor=0.2),
                RandomBrightness(factor=0.2),
            ]
        )

        # https://www.tensorflow.org/guide/keras/preprocessing_layers#preprocessing_data_before_the_model_or_inside_the_model
        return dataset.map(
            lambda x, y: (augmentation_pipeline(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

    def __modify_layers_for_transfer_learning(
        self, model: keras.Model, *, is_fine_tuning: bool
    ) -> keras.Model:
        """Private method modifying the layers of a model for transfer learning.

        This method sets the trainable attribute of the layers based on the is_fine_tuning flag.
        If the fine-tuning step takes place, all the layers of the base model will be trainable.
        If the fine-tuning step does not takes place, only the classifier layer is trainable.

        Args:
            model (keras.Model): Model to be modified for transfer learning with/without the
            fine-tuning step.
            is_fine_tuning (bool): Boolean flag determining whether the fine-tuning will be
            performed.

        Returns:
            model (keras.Model): Model modified for transfer learning.

        """
        # makes the dense classifier layer explicitly trainable
        model.layers[-1].trainable = True

        # sets all other layers to be trainable or non-trainable based on the fine tuning step
        # this has no effect on dropout or pooling layers
        for layer in model.layers[:-1]:
            layer.trainable = is_fine_tuning

        if is_fine_tuning:
            model.compile(
                optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
            )

        return model

    def load_model(self) -> keras.Model:
        """Function loading the EfficientNetB0 pre-trained model.

        This function loads the EfficientNetB0 pre-trained model on

        """
        # loading the model without the output layer (classifier)
        # setting include_top to false removes the avg pooling, dropout, and dense layers
        model_base = EfficientNetB0(
            include_top=False,
            weights="imagenet",
        )

        # defining the input tensor
        inputs = Input(shape=(224, 224, 3))
        x = model_base(inputs)

        # defining a global average pooling layer
        # it pools average value from each feature map of the model base output
        x = GlobalAveragePooling2D()(x)

        # defining a regularization dropout layer with dropout rate of 0.2
        x = Dropout(0.2)(x)

        # creating new output layer with 38 output units (specific to the problem setting)
        outputs = Dense(38, activation="softmax")(x)

        # assembling the model
        model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="plant_disease_classifier",
        )

        # compiling the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train_model(
        self, model: keras.Model, epochs: int, csv_file_path: str, *, is_fine_tuning: bool
    ) -> tuple[keras.Model, keras.callbacks.History]:
        """Public method training the model.

        This method performs the training of the CNN model. It loads the train, test and validation
        datasets. Modifies the model for the transfer learning without the fine-tuning steps. It
        trains the model using transfer learning without the fine-tuning step. Further, it modifies
        the model for the fine-tuning step and trains it. For both training stages, transfer
        learning without and with fine-tuning, it reports the accuracy and loss for each epoch.

        Args:
            model (keras.Model): Model to be trained.
            epochs (int): The number of epochs performed.
            csv_file_path (str): File path where the csv file with training metrics is saved.
            is_fine_tuning (bool): Boolean flag determining whether the fine-tuning will be
            performed.

        Returns:
            tuple[keras.Model, keras.callbacks.History]:  Tuple with trained model and training
            history.

        """
        csv_logger = CSVLogger(csv_file_path)
        model = self.__modify_layers_for_transfer_learning(model, is_fine_tuning=is_fine_tuning)

        training_history = model.fit(
            x=self.train_dataset,
            epochs=epochs,
            validation_data=self.test_dataset,
            callbacks=[csv_logger],
        )

        return model, training_history

    def evaluate_model(self, model: keras.Model) -> list[float]:
        """Public method evaluating the model.

        Args:
            model (keras.Model): Model to be trained.

        Returns:
            list[float]: List with validation metrics.

        """
        return model.evaluate(x=self.validation_dataset)


def plot_and_save_history(history: keras.callbacks.History, output_path: Path) -> None:
    """Function plotting and saving a history to the output path.

    Args:
        history (keras.callbacks.History): Object holding metrics form the training process.
        output_path (Path): Output file path, where the plot is saved.

    """
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
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

    plt.tight_layout()
    figure.savefig(output_path, format="png")
    plt.close(figure)


def save_validation_metrics(output_path: Path, metrics: dir) -> None:
    """Function saving the validation metrics as csv file.

    Args:
        output_path (Path): Output file path, where the csv is saved.
        metrics (dir): Validation metrics as directory.

    """
    with output_path.open("w") as file:
        fieldnames = ["metric", "value"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for k, v in metrics.items():
            writer.writerow({"metric": k, "value": v})


def main():
    """Main function."""
    cli = CLI()
    args = cli.parse_args()
    model_trainer = ModelTrainer(
        args.train_dataset_path, args.validation_dataset_path, args.batch_size
    )
    model = model_trainer.load_model()

    # transfer learning without the fine-tuning step
    initial_model, initial_history = model_trainer.train_model(
        model, args.epochs, str(args.output / "initial_training_metrics.csv"), is_fine_tuning=False
    )

    # plotting and saving the initial metrics without the fine-tuning step
    plot_and_save_history(initial_history, args.output / "initial_training_metrics.png")

    # validating the initial model
    validation_metrics = model_trainer.evaluate_model(initial_model)
    validation_metrics = dict(zip(initial_model.metrics_names, validation_metrics, strict=False))

    # saving the validation metrics as csv file
    save_validation_metrics(args.output / "initial_validation_metrics.csv", validation_metrics)

    # saving the initial model to the output path
    initial_model.save(args.output / "initial_model.keras")

    # transfer learning with the fine-tuning step
    final_model, final_history = model_trainer.train_model(
        initial_model,
        args.epochs,
        str(args.output / "final_training_metrics.csv"),
        is_fine_tuning=True,
    )

    # plotting and saving the final training metrics with the fine-tuning step
    plot_and_save_history(final_history, args.output / "final_training_metrics.png")

    # validating the final model
    validation_metrics = model_trainer.evaluate_model(final_model)
    validation_metrics = dict(zip(final_model.metrics_names, validation_metrics, strict=False))

    # saving the validation metrics as csv file
    save_validation_metrics(args.output / "final_validation_metrics.csv", validation_metrics)

    # saving the final model to the output path
    final_model.save(args.output / "final_model.keras")
