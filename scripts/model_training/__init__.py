"""Package training a CNN model for plant disease classification.

This package trains a EfficientNetB0 CNN architecture for plant disease classification from leaf
images. It loads the datasets and preform transfer learning without and with the fine-tuning step.
It returns plots for several metrics and the trained model. This package also provides a simple
command line interface.

"""

import logging

# logger set-up
logging.getLogger(__name__).addHandler(logging.NullHandler())
