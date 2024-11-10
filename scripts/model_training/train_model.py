"""Script training a EfficientNetB0 CNN model for plant disease classification from leaf images."""

import sys

from tensorflow import keras


def load_dataset():
    """Function to load train and test datasets."""
    # use Image generator to preprocess the images + flow_from_directory
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
    # setting include_top to false removes the avg pooling, dropout, and dense layers
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


def train_model():
    """Function to train the model."""
    return 0


if __name__ == "__main__":
    sys.exit(train_model())
