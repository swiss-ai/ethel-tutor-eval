import io
import json
import os
import tarfile
from typing import List

from utils.config import Config

import logging

import requests

from our_datasets.base_dataset import BaseDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MATH(BaseDataset):
    def __init__(self, config: Config):
        super().__init__(config)

        self._test_samples = []

    def config_name(self) -> str:
        return 'math'

    def download(self):
        dataset_path = self._config.get_dataset_path(self.config_name())

        if os.path.exists(dataset_path):
            logger.info(f"Dataset MATH already downloaded")
            return

        os.makedirs(dataset_path, exist_ok=True)

        logger.info("Downloading MATH dataset")
        dataset_url = 'https://people.eecs.berkeley.edu/~hendrycks/MATH.tar'

        response = requests.get(dataset_url)
        response.raise_for_status()

        # Extracting the tar file content
        with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:') as tar_ref:
            tar_ref.extractall(path=dataset_path)

        logger.info("Downloaded and extracted MATH dataset!")

    def load(self):
        dataset_path = os.path.join(self._config.get_dataset_path(self.config_name()), 'MATH', 'test')

        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            for file in os.listdir(category_path):
                with open(os.path.join(category_path, file), 'r') as f:
                    raw_sample = json.load(f)
                sample = {
                    'question': raw_sample['problem'],
                    'answer': raw_sample['solution'],
                    'level': raw_sample['level'],
                    'category': raw_sample['type'],
                }

                self._test_samples.append(sample)

    def get_test_samples(self) -> List:
        return self._test_samples

    def get_train_samples(self) -> List:
        return []
