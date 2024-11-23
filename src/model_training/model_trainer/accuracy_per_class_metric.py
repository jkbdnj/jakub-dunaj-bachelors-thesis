"""Nothing now."""

import tensorflow as tf
from keras.backend import epsilon
from keras.metrics import Metric
from keras.src import ops


# https://www.tensorflow.org/api_docs/python/tf/keras/Metric#add_variable
# https://www.tensorflow.org/api_docs/python/tf/Variable
# https://github.com/keras-team/keras/blob/v3.3.3/keras/src/metrics/metric.py
# https://github.com/keras-team/keras/blob/master/keras/src/metrics/reduction_metrics.py
# https://www.tensorflow.org/api_docs/python/tf/keras/Variable
# https://github.com/keras-team/keras/blob/v3.3.3/keras/src/initializers/__init__.py
# https://github.com/keras-team/keras/blob/v3.3.3/keras/src/initializers/constant_initializers.py
# https://www.tensorflow.org/guide/tensor_slicing (manipulation with tensors)
# https://github.com/keras-team/keras/blob/master/keras/src/ops/numpy.py#L322
# https://github.com/keras-team/keras/blob/v3.6.0/keras/src/ops/numpy.py#L841
# https://www.tensorflow.org/api_docs/python/tf/keras/ops
# https://www.tensorflow.org/api_docs/python/tf/is_symbolic_tensor
class AccuracyPerClassMetric(Metric):
    """Nothing now.

    Attributes:
        _total_per_class (keras.src.backend.Variable): Variable representing an array holding the
        total number of predictions per class.

    """

    def __init__(self, class_count: int, name="accuracy_per_class_metric", dtype=None):
        """Nothing now."""
        super().__init__(name=name, dtype=dtype)
        self._class_count = class_count
        self._total_per_class = self.add_variable(
            shape=(class_count,), initializer="zeros", dtype="int32", name="total_per_class"
        )

        self._correct_per_class = self.add_variable(
            shape=(class_count,), initializer="zeros", dtype="int32", name="correct_per_class"
        )

    def update_state(self, y: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """Nothing now.

        The symbolic tensors cannot be converted to a numpy array and the values cannot be accessed.
        The operations have to be done with symbolic tensors rather than with numpy arrays.

        Args:
            y (tf.Tensor): This parameter is a symbolic rank 1 tensor of shape (batch_size,)
            and contains the true labels of inputs.
            y_pred (tf.Tensor): This parameter is a symbolic rank 2 tensor of shape (batch_size,
            number_of_classes) and contains the predicted probabilities for each input for all
            classes.
            sample_weight: Nothing now.

        """
        # https://www.tensorflow.org/api_docs/python/tf/keras/ops/argmax
        # the axis is index within the shape of the tensor
        y_pred = ops.argmax(y_pred, axis=-1)
        correct_predictions = ops.equal(y, y_pred)

        for label in range(self._class_count):
            # https://www.tensorflow.org/api_docs/python/tf/keras/ops/equal
            # https://www.tensorflow.org/api_docs/python/tf/keras/ops/full
            # https://www.tensorflow.org/api_docs/python/tf/keras/ops/shape
            # https://www.tensorflow.org/api_docs/python/tf/keras/ops/reshape
            # https://www.tensorflow.org/api_docs/python/tf/keras/ops/scatter
            label_occurrences_tensor = ops.equal(y, ops.full(ops.shape(y), label))
            label_occurrences_count = ops.reshape(
                ops.sum(label_occurrences_tensor), (1,)
            )  #  ops.shape(ops.sum(label_occurrences_tensor)) == ()
            self._total_per_class.assign_add(
                ops.scatter(
                    ops.reshape(ops.convert_to_tensor(ops.array([label]), dtype="int32"), (1, 1)),
                    label_occurrences_count,
                    (self._class_count,),
                )
            )

            correct_predictions_label_tensor = ops.logical_and(
                correct_predictions, label_occurrences_tensor
            )
            correct_predictions_label_count = ops.reshape(
                ops.sum(correct_predictions_label_tensor), (1,)
            )

            self._correct_per_class.assign_add(
                ops.scatter(
                    ops.reshape(ops.convert_to_tensor(ops.array([label]), dtype="int32"), (1, 1)),
                    correct_predictions_label_count,
                    (self._class_count,),
                )
            )

    def result(self) -> tf.Tensor:
        """Nothing now."""
        return ops.divide(self._correct_per_class, self._total_per_class + epsilon())
