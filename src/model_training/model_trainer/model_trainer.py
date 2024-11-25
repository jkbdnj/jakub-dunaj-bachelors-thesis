"""Module providing training functionalities to train EfficientNetB0 CNN model.

This module provides training functionalities to train EfficientNetB0 CNN model for plant disease
classification from leaf images.

"""

import logging
from pathlib import Path

import keras
import tensorflow as tf
from keras.applications import EfficientNetB0
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
from keras.optimizers import RMSprop
from keras.optimizers.schedules import ExponentialDecay
from keras.utils import image_dataset_from_directory

from model_trainer import utils

from .accuracy_per_class_metric import AccuracyPerClassMetric
from .cli import CLI
from .custom_history_callback import CustomHistory
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

    def __init__(
        self, train_dataset_path: Path, validation_dataset_path: Path, batch_size: int, epochs: int
    ):
        """Constructor instantiating an object of the ModelTrainer class.

        This constructor loads up the train, test and validation datasets and applies augmentation
        pipeline to the train dataset. It also sets up the instance variables train_dataset,
        test_dataset and validation_dataset.

        Args:
            train_dataset_path (Path): The path to the train dataset.
            validation_dataset_path (Path): The path to the validation dataset.
            batch_size (int): The batch size used.
            epochs (int): Number of epoch within one training process.

        """
        train_dataset, test_dataset, validation_dataset = self._load_dataset(
            train_dataset_path, validation_dataset_path, batch_size
        )
        self._train_dataset = self._apply_augmenting_pipeline(train_dataset)
        self._test_dataset = test_dataset
        self._validation_dataset = validation_dataset
        self._batch_size = batch_size
        self._epochs = epochs
        self._labels = train_dataset.class_names

    @property
    def labels(self):
        """Property returning labels as list."""
        return self._labels

    def _load_dataset(
        self, train_dataset_path: Path, validation_dataset_path: Path, batch_size: int
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Protected method loading train, test and validation datasets.

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

        if not (
            train_dataset.class_names == test_dataset.class_names == validation_dataset.class_names
        ):
            error_message = "Train, test, and validation datasets differ in the class name orders!"
            raise DatasetError(error_message)

        return train_dataset, test_dataset, validation_dataset

    def _apply_augmenting_pipeline(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Protected method applying augmentation operations to dataset images.

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
                RandomRotation(factor=0.1, fill_mode="reflect"),
                RandomContrast(factor=0.2),
                RandomBrightness(factor=0.2),
            ]
        )

        # https://www.tensorflow.org/guide/keras/preprocessing_layers#preprocessing_data_before_the_model_or_inside_the_model
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map
        return dataset.map(
            lambda x, y: (augmentation_pipeline(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

    def _compile_model(self, model: keras.Model) -> None:
        """Protected method compiling the model.

        This method sets up the optimizer and compiles the model.

        Args:
            model (keras.Model): The model to compile.

        """
        steps_per_epoch = int(len(self._train_dataset))
        decay_steps = steps_per_epoch * 2
        learning_rate_schedule = ExponentialDecay(
            initial_learning_rate=0.001, decay_steps=decay_steps, decay_rate=0.90
        )
        model.compile(
            optimizer=RMSprop(
                learning_rate=learning_rate_schedule, rho=0.9, momentum=0.9, weight_decay=1e-5
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", AccuracyPerClassMetric(len(self.labels))],
        )

    def _modify_layers_for_transfer_learning(
        self, model: keras.Model, *, is_fine_tuning: bool
    ) -> keras.Model:
        """Protected method modifying the layers of a model for transfer learning.

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
            self._compile_model(model)

        return model

    def load_model(self) -> keras.Model:
        """Public method loading the EfficientNetB0 pre-trained model.

        This method loads a EfficientNetB0 model. The EfficientNetB0 model is pre-trained
        on ImageNet dataset and is loaded without the classifier layer. A new classifier
        layer with 38 outputs and softmax activation function is added to the architecture.

        Returns:
            (keras.Model): The loaded EfficientNetB0 model.

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

        self._compile_model(model)

        return model

    def train_model(
        self, model: keras.Model, *, is_fine_tuning: bool
    ) -> tuple[keras.Model, keras.callbacks.History]:
        """Public method training the model.

        This method performs the training of the CNN model. It loads the train, test and validation
        datasets. Modifies the model for the transfer learning without the fine-tuning steps. It
        trains the model using transfer learning without the fine-tuning step. Further, it modifies
        the model for the fine-tuning step and trains it. For both training stages, transfer
        learning without and with fine-tuning, it reports the accuracy and loss for each epoch.

        Args:
            model (keras.Model): Model to be trained.
            csv_file_path (str): File path where the csv file with training metrics is saved.
            is_fine_tuning (bool): Boolean flag determining whether the fine-tuning will be
            performed.

        Returns:
            tuple[keras.Model, keras.callbacks.History]:  Tuple with trained model and training
            history.

        """
        model = self._modify_layers_for_transfer_learning(model, is_fine_tuning=is_fine_tuning)

        training_history = model.fit(
            x=self._train_dataset,
            epochs=self._epochs,
            callbacks=[CustomHistory()],
            validation_data=self._test_dataset,
        )

        return model, training_history

    def evaluate_model(self, model: keras.Model) -> list[float]:
        """Public method evaluating the model.

        Args:
            model (keras.Model): Model to be trained.

        Returns:
            list[float]: List with validation metrics.

        """
        return model.evaluate(x=self._validation_dataset)


def main():
    """Main function."""
    cli = CLI()
    args = cli.parse_args()
    model_trainer = ModelTrainer(
        args.train_dataset_path, args.validation_dataset_path, args.batch_size, args.epochs
    )
    model = model_trainer.load_model()

    # transfer learning without the fine-tuning step
    initial_model, initial_history = model_trainer.train_model(model, is_fine_tuning=False)

    # plotting and saving the initial metrics without the fine-tuning step
    utils.save_accuracy_and_loss_as_plot(initial_history, args.output, keyword="initial")
    utils.save_accuracy_per_class_as_plot(
        initial_history, model_trainer.labels, args.output, keyword="initial"
    )
    utils.save_history_as_json(
        initial_history, args.output, model_trainer.labels, keyword="initial"
    )

    # validating the initial model
    evaluation_metrics = model_trainer.evaluate_model(initial_model)
    evaluation_metrics = dict(zip(initial_model.metrics_names, evaluation_metrics, strict=False))
    utils.save_evaluation_metrics_as_json(evaluation_metrics, args.output, keyword="initial")

    # saving the initial model to the output path
    initial_model.save(args.output / "initial_model.keras")

    # transfer learning with the fine-tuning step
    final_model, final_history = model_trainer.train_model(
        initial_model,
        is_fine_tuning=True,
    )

    # plotting and saving the final metrics with the fine-tuning step
    utils.save_accuracy_and_loss_as_plot(final_history, args.output, keyword="final")
    utils.save_accuracy_per_class_as_plot(
        final_history, model_trainer.labels, args.output, keyword="final"
    )
    utils.save_history_as_json(final_history, args.output, model_trainer.labels, keyword="final")

    # validating the final model
    evaluation_metrics = model_trainer.evaluate_model(final_model)
    evaluation_metrics = dict(zip(final_model.metrics_names, evaluation_metrics, strict=False))
    utils.save_evaluation_metrics_as_json(evaluation_metrics, args.output, keyword="final")

    # saving the final model to the output path
    final_model.save(args.output / "final_model.keras")
