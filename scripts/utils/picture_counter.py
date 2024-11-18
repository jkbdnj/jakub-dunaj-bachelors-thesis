"""Utility module to count images in the three dataset subsets."""

import json
from pathlib import Path

# constant for the path to the color subset of the initial dataset
COLOR_PATH = Path("PATH_TO_SUBSET/color")

# constant for the path to the segmented subset of the initial dataset
SEGMENTED_PATH = Path("PATH_TO_SUBSET/segmented")

# constant for the path to the artificial background subset of the initial dataset
ARTIFICIAL_PATH = Path("PATH_TO_SUBSET/artificial_background")

dataset_paths = [
    COLOR_PATH,
    SEGMENTED_PATH,
    ARTIFICIAL_PATH,
]

dictionary = {}
aggregated_dict = {}
for dataset_path in dataset_paths:
    subset_dict = {}
    for class_path in dataset_path.iterdir():
        if class_path.is_dir():
            length = len(list(class_path.glob("*.[jJ][pP][gG]")))
            subset_dict[class_path.name] = length

            if class_path.name not in aggregated_dict:
                aggregated_dict[class_path.name] = length
            else:
                aggregated_dict[class_path.name] += length

    dictionary[dataset_path.name] = subset_dict

dictionary["aggregated"] = aggregated_dict

with Path("./counts.json").open("w") as file:
    json.dump(dictionary, file, indent=4)
