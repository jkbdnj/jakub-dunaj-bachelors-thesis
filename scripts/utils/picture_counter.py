"""Utility module counting images in a dataset with subsets that contain classes."""

import json
from pathlib import Path

DATASET_PATH = Path("/Users/kubkodunaj/Desktop/jakub-dunaj-bachelors-thesis/datasets/final_dataset")

dictionary = {}
aggregated_dictionary = {}
for subset_path in DATASET_PATH.iterdir():
    if subset_path.is_dir():
        subset_dictionary = {}
        for class_path in subset_path.iterdir():
            if class_path.is_dir():
                length = len(list(class_path.glob("*.[jJ][pP][gG]")))
                subset_dictionary[class_path.name] = length

                if class_path.name not in aggregated_dictionary:
                    aggregated_dictionary[class_path.name] = length
                else:
                    aggregated_dictionary[class_path.name] += length
        subset_dictionary = dict(sorted(subset_dictionary.items()))
        subset_dictionary["total"] = sum(subset_dictionary.values())
        dictionary[subset_path.name] = subset_dictionary

aggregated_dictionary = dict(sorted(aggregated_dictionary.items()))
aggregated_dictionary["total"] = sum(aggregated_dictionary.values())
dictionary["aggregated"] = aggregated_dictionary

with Path("./initial_counts.json").open("w") as file:
    json.dump({DATASET_PATH.name: dictionary}, file, indent=4)
