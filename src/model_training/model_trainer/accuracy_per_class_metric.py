"""Module providing custom accuracy per class metric."""

import tensorflow as tf
from keras.backend import epsilon
from keras.metrics import Metric
from keras.src import ops


class AccuracyPerClassMetric(Metric):
    """Custom metric class computing the accuracy per class during the training process.

    This custom metric extends the Metric class and has the functionality to compute
    the accuracy per class during the training process. It tracks two variable holding
    the total number of occurrences per class/label and the number of correct predictions.
    The methods of this class work with symbolic tensors.

    Useful links for implementation:
        https://keras.io/api/metrics/base_metric/#metric-class
        https://www.tensorflow.org/api_docs/python/tf/Variable
        https://github.com/keras-team/keras/blob/v3.3.3/keras/src/metrics/metric.py
        https://github.com/keras-team/keras/blob/master/keras/src/metrics/reduction_metrics.py#L97
        https://www.tensorflow.org/api_docs/python/tf/keras/ops
        https://www.tensorflow.org/api_docs/python/tf/keras/ops/argmax
        https://www.tensorflow.org/api_docs/python/tf/keras/ops/equal
        https://www.tensorflow.org/api_docs/python/tf/keras/ops/full
        https://www.tensorflow.org/api_docs/python/tf/keras/ops/shape
        https://www.tensorflow.org/api_docs/python/tf/keras/ops/reshape
        https://www.tensorflow.org/api_docs/python/tf/keras/ops/scatter

    Attributes:
        _class_count (int): The number of classes the model is learning.
        _total_per_class (keras.src.backend.Variable): Variable representing an array holding the
        total number of predictions per class.
        _correct_per_class (keras.src.backend.Variable): Variable representing an array holding the
        correct number of predictions per class.

    """

    def __init__(self, class_count: int, name="accuracy_per_class", dtype=None):
        """Constructor instantiating an object of the AccuracyPerClassMetric class."""
        super().__init__(name=name, dtype=dtype)
        self._class_count = class_count
        self._total_per_class = self.add_variable(
            shape=(class_count,), initializer="zeros", dtype="int32", name="total_per_class"
        )

        self._correct_per_class = self.add_variable(
            shape=(class_count,), initializer="zeros", dtype="int32", name="correct_per_class"
        )

    def update_state(self, y: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """Function updating the _total_per_class and _correct_per_class instance attributes.

        This method accumulates the statistics for the accuracy per class metric.The symbolic
        tensors cannot be converted to a numpy array and the values cannot be accessed. The
        operations have to be done with symbolic tensors rather than with numpy arrays. The
        reset_state() method is implemented by the superclass Metric.

        Args:
            y (tf.Tensor): This parameter is a symbolic 1D (rank 1) tensor of shape (batch_size,)
            and contains the true labels of inputs.
            y_pred (tf.Tensor): This parameter is a symbolic 2D (rank 2) tensor of shape
            (batch_size, number_of_classes) and contains the predicted probabilities for
            each input for all classes.
            sample_weight: Weights for batch elements. It is ignored.

        """
        # axis is index within the shape of a tensor (y_pred is a 2D tensor)
        label_predictions = ops.argmax(y_pred, axis=-1)

        # returns 1D tensor containing correct predictions per batch element
        correct_predictions = ops.equal(y, label_predictions)

        for label in range(self._class_count):
            # returns 1D tensor containing 1s at indices the label is at in y tensor
            label_flags = ops.equal(y, ops.full(ops.shape(y), label))

            # returns 1D tensor containing the number of occurrences of label in batch
            label_count = ops.reshape(ops.sum(label_flags), (1,))

            # assigns a tensor to variable
            # tensor contains the number of occurrences of a label at index==label
            self._total_per_class.assign_add(
                # returns a 1D tensor of length self._class_count
                # label_count will be at index given by label
                ops.scatter(
                    ops.reshape(ops.convert_to_tensor(ops.array(label), dtype="int32"), (1, 1)),
                    label_count,
                    (self._class_count,),
                )
            )

            # makes logical and between correct_predictions and label_flags
            # returns tensor containing 1s where the prediction for label is correct
            correct_label_flags = ops.logical_and(correct_predictions, label_flags)

            # returns 1D tensor containing the number of correct predictions of label in batch
            correct_label_count = ops.reshape(ops.sum(correct_label_flags), (1,))

            # assigns a tensor to variable
            # tensor contains correct number of predictions of a label at the index==label
            self._correct_per_class.assign_add(
                # returns 1D tensor of length self._class_count
                # the correct_label_count will be at index given by label
                ops.scatter(
                    ops.reshape(ops.convert_to_tensor(ops.array(label), dtype="int32"), (1, 1)),
                    correct_label_count,
                    (self._class_count,),
                )
            )

    def result(self):
        """Function computing current metric value."""
        return ops.divide(self._correct_per_class, self._total_per_class + epsilon())
