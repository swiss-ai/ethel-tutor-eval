import io
import os
import logging
import shutil
import zipfile
from typing import List
from datasets import load_dataset, load_from_disk, concatenate_datasets
import requests
import pandas as pd

from our_datasets.base_dataset import BaseDataset
from utils.config import Config
from utils.read_utils import read_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MGSM_DE(BaseDataset):
    def __init__(self, config: Config):
        super().__init__(config)
        self._train_samples = []
        self._test_samples = []

        self._train_idx_shuffle = []

    def config_name(self) -> str:
        return "mgsm"

    def download(self):
        dataset_path = self._config.get_dataset_path(self.config_name())

        if os.path.exists(dataset_path):
            logger.info(f"Dataset MGSM_DE already downloaded")
            return

        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "de"), exist_ok=True)

        logger.info("Downloading MGSM_DE dataset")

        de = load_dataset("juletxara/mgsm", "de")

        de.save_to_disk(os.path.join(dataset_path, "de"))

        logger.info("Downloaded MGSM_DE dataset!")

    def load(self):
        dataset_path = self._config.get_dataset_path(self.config_name())
        de_samples = load_from_disk(os.path.join(dataset_path, "de"))

        train_de = de_samples["train"].to_pandas()
        train_de["language"] = "de"
        self._train_samples = train_de.to_dict(orient="records")

        test_de = de_samples["test"].to_pandas()
        test_de["language"] = "de"

        test_dataframes = test_de

        self._test_samples = test_dataframes.to_dict(orient="records")

    def get_test_samples(self) -> List:
        return self._test_samples

    def get_train_samples(self) -> List:
        return self._train_samples


class MGSM_FR(BaseDataset):
    def __init__(self, config: Config):
        super().__init__(config)
        self._train_samples = []
        self._test_samples = []

        self._train_idx_shuffle = []

    def config_name(self) -> str:
        return "mgsm"

    def download(self):
        dataset_path = self._config.get_dataset_path(self.config_name())

        if os.path.exists(dataset_path):
            logger.info(f"Dataset MGSM_FR already downloaded")
            return

        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_path, "fr"), exist_ok=True)

        logger.info("Downloading MGSM_FR dataset")

        fr = load_dataset("juletxara/mgsm", "fr")

        fr.save_to_disk(os.path.join(dataset_path, "fr"))

        logger.info("Downloaded MGSM_FR dataset!")

    def load(self):
        dataset_path = self._config.get_dataset_path(self.config_name())
        fr_samples = load_from_disk(os.path.join(dataset_path, "fr"))

        train_fr = fr_samples["train"].to_pandas()
        train_fr["language"] = "fr"
        self._train_samples = train_fr.to_dict(orient="records")

        test_fr = fr_samples["test"].to_pandas()
        test_fr["language"] = "fr"

        test_dataframes = test_fr

        self._test_samples = test_dataframes.to_dict(orient="records")

    def get_test_samples(self) -> List:
        return self._test_samples

    def get_train_samples(self) -> List:
        return self._train_samples
