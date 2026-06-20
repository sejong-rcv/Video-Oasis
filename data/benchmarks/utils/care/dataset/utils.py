import json
def load_dataset_config(config_path, dataset_name):
    with open(config_path) as f:
        data_config = json.load(f)[dataset_name]
    return data_config