"""Module providing custom history callback.

The CustomHistory class logs additionally the epoch duration in seconds.

"""

import time

from keras.callbacks import History


class CustomHistory(History):
    """Custom class extending the History class.

    This custom history class extends the keras callback History by
    logging the epoch duration.

    Useful links for implementation:
        https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History
        https://github.com/keras-team/keras/blob/v3.3.3/keras/src/callbacks/callback.py

    Attributes:
        _epoch_start_time (float): Float number holding the start time of the epoch execution.

    """

    def __init__(self):
        """Constructor instantiating an object of the CustomHistory class."""
        super().__init__()

    def on_epoch_begin(self, *args):
        """Method initiating the time measurement at the start of an epoch.

        Args:
            *args: Stands for epoch and logs positional arguments. These are ignored
            in this implementation.

        """
        self._epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        """Method logging epoch duration in second at the end of an epoch.

        Args:
            epoch (int): Index of an epoch.
            logs (dict | None): Directory with results for the training and validation metrics
            during the epoch.

        """
        logs = logs or {}
        epoch_end_time = time.time()
        logs["epoch_duration_sec"] = epoch_end_time - self._epoch_start_time
        super().on_epoch_end(epoch, logs)
