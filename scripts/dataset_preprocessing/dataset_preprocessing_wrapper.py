"""Empty for now."""

import logging
import sys

logger = logging.getLogger("wrapper")
FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="dataset_preprocessing.log", format=FORMAT, level=logging.INFO)


if __name__ == "__main__":
    sys.exit(1)
