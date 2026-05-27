import re

import os
import re
from pathlib import Path
import json

def find_repo_path():
    """
    Find the path to a repository based on .git directory.
    """
    file_path = Path(__file__).resolve()
    while file_path != file_path.parent:
        if (file_path / ".git").exists():
            return file_path
        file_path = file_path.parent
    raise FileNotFoundError("No .git directory found in the path hierarchy.")


REPO_PATH = find_repo_path()
gem_data_path = 'your_path_to_gem_data'  # TODO: set your path to gem data



GEM = {
    "annotation_path": os.path.join(REPO_PATH, "GEM-250K.jsonl"),  # TODO: set your path to GEM-250K jsonl file
    "data_path": gem_data_path,
    "data_source": "gem",
}


data_dict = {
    "gem": GEM
}

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = [
        "gem",
    ]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
