"""Nothing now."""

from keras.metrics import Metric


# https://www.tensorflow.org/api_docs/python/tf/keras/Variable
# https://github.com/keras-team/keras/blob/v3.3.3/keras/src/initializers/__init__.py
# https://github.com/keras-team/keras/blob/v3.3.3/keras/src/initializers/constant_initializers.py
class AccuracyPerClassMetric(Metric):
    """Nothing now.

    Attributes:
        _total_per_class (keras.src.backend.Variable): Variable representing an array holding the
        total number of predictions per class. The

    """

    def __init__(self, class_count: int, name="accuracy_per_class_metric", dtype=None):
        """Nothing now."""
        # dtype is float32 per default (result of the metric)
        super().__init__(name=name, dtype=dtype)
        self._total_per_class = self.add_variable(
            shape=(class_count), initializer="zeros", dtype="int", name="total_per_class"
        )

        self._correct_per_class = self.add_variable(
            shape=(class_count), initializer="zeros", dtype="int", name="correct_per_class"
        )

    def update_state():
        """Nothing now."""

    def result() -> list[float]:
        """Nothing now."""
