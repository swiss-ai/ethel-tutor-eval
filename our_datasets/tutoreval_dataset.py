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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TutorEval(BaseDataset):
    def __init__(self, config: Config):
        super().__init__(config)
        self._test_samples = []

    def config_name(self) -> str:
        return "TutorEval"

    def download(self):
        logger.info(self.config_name())
        dataset_path = self._config.get_dataset_path(self.config_name())

        if os.path.exists(dataset_path):
            logger.info(f"Dataset TutorEval already downloaded")
            return

        os.makedirs(dataset_path, exist_ok=True)

        logger.info("Downloading TutorEval dataset")
        data = load_dataset("princeton-nlp/TutorEval")["train"]
        data.save_to_disk(dataset_path)

        logger.info("Downloaded TutorEval dataset!")

    def load(self):
        dataset_path = self._config.get_dataset_path(self.config_name())
        self._test_samples = load_from_disk(dataset_path)
        # self._train_samples = read_jsonl(os.path.join(dataset_path, 'train.jsonl'))

    def get_test_samples(self) -> List:
        return self._test_samples

    def get_train_samples(self) -> List:
        return []

    def choose_book(self, closed_book: bool = True):
        self._test_samples = self._test_samples.filter(
            lambda x: x["closed_book"] == bool(closed_book)
        )
