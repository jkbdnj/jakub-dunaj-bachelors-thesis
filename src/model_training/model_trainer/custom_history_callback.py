"""Module providing custom history callback.

The CustomHistory class logs additionally the epoch duration in seconds.

"""

import time

from keras.callbacks import History


# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History
# https://github.com/keras-team/keras/blob/v3.3.3/keras/src/callbacks/callback.py
class CustomHistory(History):
    """Custom class extending the History class.

    This custom history class extends the keras callback History by
    logging the epoch duration.

    Attributes:
        _epoch_start_time (float): Float number holding the start time of the epoch execution.

    """

    def __init__(self):
        """Constructor instantiating an object of the CustomHistory class."""
        super().__init__()

    def on_epoch_begin(self, *args):
        """Method initiating the time measurement at the start of an epoch."""
        self._epoch_start_time: float = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """Method logging the epoch duration in second at the end of an epoch."""
        logs = logs or {}
        epoch_end_time = time.time()
        logs["epoch_duration_sec"] = epoch_end_time - self._epoch_start_time
        super().on_epoch_end(epoch, logs)
