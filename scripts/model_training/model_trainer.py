"""Script training a EfficientNetB0 CNN model for plant disease classification from leaf images."""

import logging
from pathlib import Path

import tensorflow as tf
from cli import parse_args
from exceptions import DatasetError
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
from keras.utils import image_dataset_from_directory
from tf import keras

logger = logging.getLogger("root." + __name__)


def apply_augmenting_pipeline(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Function applying augmentation operations to dataset images.

    This function uses an augmentation pipeline with 4 augmentation layers. These layers are:
    RandomFlip, RandomRotation, RandomContrast, and RandomBrightness. Each layer applies an
    augmentation operation. The randomness lies in the range of possible operation configurations
    (e.g. the range of brightness increase or decrease).

    Setting the fill_mode as constant inn RandomRotation adds will fill the missing information
    with 0s. This adds the least artificial information that might otherwise add unwanted distortion
    to the dataset. Setting the factor to 0.2 in RandomContrast will change the contrast of a pixel
    only slightly. The same goes for RandomBrightness.

    Args:
        dataset (tf.data.Dataset): Dataset that will be augmented using the augmentation pipeline.

    Returns:
        tf.data.Dataset: Augmented dataset by the defined augmentation pipeline.

    """
    augmentation_pipeline = Pipeline(
        [
            RandomFlip(mode="horizontal_and_vertical"),
            RandomRotation(fill_mode="constant"),
            RandomContrast(factor=0.2),
            RandomBrightness(factor=0.2),
        ]
    )

    return dataset.map(augmentation_pipeline, num_parallel_calls=tf.Data.AUTOTUNE)


def load_dataset(
    train_dataset_path: Path, test_dataset_path: Path, batch_size: int
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Function loading train and test datasets.

    This function loads the training and testing datasets from the final dataset.

    Args:
        train_dataset_path (str): The path to the train dataset.
        test_dataset_path (str):  The path to the test dataset.
        batch_size: The batch size.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset]: Tuple with train and test datasets.

    """
    train_dataset = image_dataset_from_directory(
        train_dataset_path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(224, 224),
        interpolation="bilinear",
    )

    test_dataset = image_dataset_from_directory(
        test_dataset_path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(224, 224),
        interpolation="bilinear",
    )

    if set(train_dataset.class_names) != set(test_dataset.class_names):
        error_message = "Train and tests datasets differ in the class names!"
        raise DatasetError(error_message)

    return train_dataset, test_dataset


def load_model():
    """Function loading the EfficientNetB0 pre-trained model.

    This function loads the EfficientNetB0 pre-trained model on

    """
    # loading the model without the output layer (classifier)
    # setting include_top to false removes the avg pooling, dropout, and dense layers of the model
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

    # defining a regularization dropout layer with dropout rate of 0.2 (as original architecture)
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


def modify_layers_for_transfer_learning(model, *, is_fine_tuning: bool):
    """Function modifying the layers of a model for transfer learning.

    This function sets the trainable attribute of the layers either to True or False depending on
    the fine-tuning step. If the fine-tuning step takes place, all the layers of the base model will
    be trainable. If the fine-tuning step does not takes place, the layers of the model base will be
    non-trainable. The last dense classifier layer is always trainable.

    Args:
        model: Model to be modified for transfer learning with/without the fine-tuning step.
        is_fine_tuning (bool): Bool variable defining if the fine-tuning step will be applied.

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


def train_model(args):
    """Function training the model.

    This function performs the training of the CNN model. It loads the train and test datasets.
    Modifies the model for the transfer learning without the fine-tuning steps. It trains the model
    using transfer learning without the fine-tuning step. Further, it modifies the model for the
    fine-tuning step and trains it. For both training stages, transfer learning without and with
    fine-tuning, it reports the accuracy and loss for each epoch.

    Args:
        args: Object holding the parsed arguments.

    """
    train_dataset, test_dataset = load_dataset(
        args.train_dataset_path, args.test_dataset_path, args.batch_size
    )
    model = load_model()

    # transfer learning without the fine-tuning step
    model = modify_layers_for_transfer_learning(model, is_fine_tuning=False)
    model.fit(x=test_dataset, epochs=args.epochs, validation_data=train_dataset)

    # transfer learning with the fine-tuning step
    # model = modify_layers_for_transfer_learning(model, is_fine_tuning=True)
    # history = model.fit(x=train_dataset, epochs=args.epochs, validation_data=test_dataset)

    return 0


def plot_and_save_history(output_path: Path):
    """Function plotting and saving a history to the output path."""
    return output_path


def main():
    """Main function."""
    parse_args()
    # sys.exit(train_model(args))


if __name__ == "__main__":
    main()
