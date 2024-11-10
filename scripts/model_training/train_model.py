"""Script training a EfficientNetB0 CNN model for plant disease classification from leaf images."""

import sys

from tensorflow import keras


def load_dataset():
    """Function loading train and test datasets."""
    train_dataset = keras.utils.image_dataset_from_directory(
        "/Users/kubkodunaj/Desktop/jakub-dunaj-bachelors-thesis/datasets/final_dataset/train",
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=32,
        image_size=(224, 224),
        interpolation="bilinear",
    )

    test_dataset = keras.utils.image_dataset_from_directory(
        "/Users/kubkodunaj/Desktop/jakub-dunaj-bachelors-thesis/datasets/final_dataset/test",
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=32,
        image_size=(224, 224),
        interpolation="bilinear",
    )

    return train_dataset, test_dataset


def load_model():
    """Function to load the EfficientNetB0 pre-trained model."""
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
    model = keras.Model(inputs=inputs, outputs=outputs, name="plant_disease_analyzer")

    # compiling the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def modify_layers_for_transfer_learning(model, *, is_fine_tuning: bool):
    """Function modifying the layers of a model for transfer learning.

    This function sets the trainable attribute of the layers either to True or False depending on
    the fine-tuning step. If the fine-tuning step takes place, all the layers of the base model will
    be trainable. If the fine-tuning step does not takes place, the layers of the model base will be
    non-trainable. The last dense classifier layer is always trainable.

    Args:
        model: Model whose layers except the classifier will be set non-trainable.
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


def train_model():
    """Function training the model.

    This function performs the training of the CNN model. It loads the train and test datasets.
    Modifies the model for the transfer learning without the fine-tuning steps. It trains the model
    using transfer learning without the fine-tuning step. Further, it modifies the model for the
    fine-tuning step and trains it. For both training stages, transfer learning without and with
    fine-tuning, it reports the accuracy and loss for each epoch.
    """
    train_dataset, test_dataset = load_dataset()
    model = load_model()
    model = modify_layers_for_transfer_learning(model, is_fine_tuning=False)
    model.fit(x=train_dataset, epochs=15, validation_data=test_dataset)
    model = modify_layers_for_transfer_learning(model, is_fine_tuning=True)
    model.fit(x=train_dataset, epochs=15, validation_data=test_dataset)
    return 0


if __name__ == "__main__":
    sys.exit(train_model())
