import io
import os
import logging
import shutil
import zipfile
from typing import List

import requests

from datasets.base_dataset import BaseDataset
from utils.config import Config
from utils.read_utils import read_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GSM8K(BaseDataset):
    def __init__(self, config: Config):
        super().__init__(config)
        self._train_samples = []
        self._test_samples = []

        # store idx shuffle, unique for the instance, to use as n-shot learning examples
        self._train_idx_shuffle = []

    def config_name(self) -> str:
        return "gsm8k"

    def download(self):
        dataset_path = self._config.get_dataset_path(self.config_name())

        if os.path.exists(dataset_path):
            logger.info(f"Dataset GSM8k already downloaded")
            return

        os.makedirs(dataset_path, exist_ok=True)

        logger.info("Downloading GSM8k dataset")
        dataset_url = 'https://github.com/openai/grade-school-math/archive/refs/heads/master.zip'

        response = requests.get(dataset_url)
        response.raise_for_status()

        # Extracting the zip file content
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            for member in zip_ref.namelist():
                # We need to extract only files from the data directory
                if member.startswith(os.path.join('grade-school-math-master', 'grade_school_math', 'data') + os.sep):
                    if '.' not in member.split(os.sep)[-1]:
                        continue  # ignore /data/ directory
                    # This will extract the file into the desired directory structure
                    zip_ref.extract(member, dataset_path)
                    # Rename the extracted files to remove the prefix directories
                    extracted_path = os.path.join(dataset_path, member)
                    target_path = os.path.join(dataset_path, os.path.basename(member))
                    shutil.move(extracted_path, target_path)

        extracted_main_dir = os.path.join(dataset_path, 'grade-school-math-master')
        if os.path.exists(extracted_main_dir):
            shutil.rmtree(extracted_main_dir)

        logger.info("Downloaded GSM8k dataset!")

    def load(self):
        dataset_path = self._config.get_dataset_path(self.config_name())
        self._test_samples = read_jsonl(os.path.join(dataset_path, 'test.jsonl'))
        self._train_samples = read_jsonl(os.path.join(dataset_path, 'train.jsonl'))

    def get_test_samples(self) -> List:
        return self._test_samples

    def get_train_samples(self) -> List:
        return self._train_samples
