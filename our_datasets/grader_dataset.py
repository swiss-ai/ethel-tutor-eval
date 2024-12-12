import io
import os
import logging
import shutil
import zipfile
from typing import List
from datasets import load_dataset, load_from_disk
import requests

from our_datasets.base_dataset import BaseDataset
from utils.config import Config
from utils.read_utils import read_jsonl
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Grader(BaseDataset):
    def __init__(self, config: Config):
        super().__init__(config)
        self._test_samples = []

    def config_name(self) -> str:
        return "Grader"

    def download(self):
        logger.info(self.config_name())
        dataset_path = self._config.get_dataset_path(self.config_name())

        if os.path.exists(dataset_path):
            logger.info(f"Dataset TutorEval already downloaded")
            return
        else:
            raise FileNotFoundError("Grader dataset not found")

    def load(self):
        dataset_path = self._config.get_dataset_path(self.config_name())
        with open(dataset_path +'result_samples_to_grade.json') as f:
            json_file = json.load(f)
        self._test_samples = [item for item in json_file]

    def get_test_samples(self) -> List:
        return self._test_samples

    def get_train_samples(self) -> List:
        return []

    def choose_book(self, closed_book: bool = True):
        self._test_samples = [json_item for json_item in self._test_samples if json_item.get('closed_book') == bool(closed_book)]