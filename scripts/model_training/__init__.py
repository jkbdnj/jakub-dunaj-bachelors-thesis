"""Package training a CNN model for plant disease classification.

This package trains a EfficientNetB0 CNN architecture for plant disease classification from leaf
images. It loads the datasets and preform transfer learning without and with the fine-tuning step.
It returns plots for several metrics and the trained model. This package also provides a simple
command line interface.

"""

import logging

# logger set-up
logger = logging.getLogger("root")
FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="model_training.log", format=FORMAT, level=logging.INFO)
