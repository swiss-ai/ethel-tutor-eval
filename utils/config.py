import os

import yaml


class Config:
    def __init__(self, config_path: str, dataset_dir: str):
        self._config = self._load_yaml_config(config_path)
        self._dataset_dir = dataset_dir

    def _load_yaml_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_dataset_path(self, dataset_name: str) -> str:
        return os.path.join(self._dataset_dir, self._config['our_datasets'][dataset_name.lower()])

    def get_records_path(self) -> str:
        return self._config['record_dir']
