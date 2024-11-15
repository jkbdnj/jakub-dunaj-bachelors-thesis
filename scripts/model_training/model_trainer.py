"""Script training a EfficientNetB0 CNN model for plant disease classification from leaf images."""

import sys
from pathlib import Path

from cli import parse_args
from exceptions import DatasetException
from tensorflow import keras

# script_name = __file__.stem
# logger = logging.getLogger(script_name)
# FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
# logging.basicConfig(filename=__file__.name + ".log", format=FORMAT, level=logging.INFO)


def load_dataset(train_dataset_path: Path, test_dataset_path: Path, batch_size):
    """Function loading train and test datasets.

    This function loads the training and testing datasets from the final dataset.

    Args:
        train_dataset_path (str): The path to the train dataset.
        test_dataset_path (str):  The path to the test dataset.
        batch_size: The batch size.

    """
    train_dataset = keras.utils.image_dataset_from_directory(
        train_dataset_path,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(224, 224),
        interpolation="bilinear",
    )

    test_dataset = keras.utils.image_dataset_from_directory(
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
        raise DatasetException(error_message)

    return train_dataset, test_dataset


def load_model():
    """Function loading the EfficientNetB0 pre-trained model.

    This function loads the EfficientNetB0 pre-trained model on

    """
    # loading the model without the output layer (classifier)
    # setting include_top to false removes the avg pooling, dropout, and dense layers of the model
    model_base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
    )

    # defining the input tensor
    inputs = keras.layers.Input(shape=(224, 224, 3))
    x = model_base(inputs)

    # defining a global average pooling layer
    # it pools average value from each feature map of the model base output
    x = keras.layers.GlobalAveragePooling2D()(x)

    # defining a regularization dropout layer with dropout rate of 0.2 (as original architecture)
    x = keras.layers.Dropout(0.2)(x)

    # creating new output layer with 38 output units (specific to the problem setting)
    outputs = keras.layers.Dense(38, activation="softmax")(x)

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


def plot_and_save_history():
    """Function plotting and saving a history to the output path."""
    return 0


def main():
    """Main function."""
    args = parse_args()
    sys.exit(train_model(args))


if __name__ == "__main__":
    main()