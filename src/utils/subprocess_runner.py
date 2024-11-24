"""Simple module running the model-trainer as subprocess."""

import subprocess
import time

TRAIN_DATASET = "PATH_TO/datasets/final_dataset/train"
VALIDATION_DATASET = "PATH_TO/datasets/final_dataset/validation"
OUTPUT_PATH = "PATH_TO_OUTPUT"
BATCH_SIZE = 32
EPOCHS = 8

commands = [
    "model-trainer",
    TRAIN_DATASET,
    VALIDATION_DATASET,
    "-o",
    OUTPUT_PATH,
    "-b",
    str(BATCH_SIZE),
    "-e",
    str(EPOCHS),
]

start_point = time.time()

# running the training process
subprocess.run(commands, check=False)

end_point = time.time()

running_time = time.strftime("%Hh-%Mm-%Ss", time.gmtime(end_point - start_point))
