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


class MGSM(BaseDataset):
    def __init__(self, config: Config):
        super().__init__(config)
        self._train_samples_de = []
        self._train_samples_fr = []
        self._test_samples = []

        self._train_idx_shuffle = []

    def config_name(self) -> str:
        return "mgsm"

    def download(self):
        dataset_path = self._config.get_dataset_path(self.config_name())

        if os.path.exists(dataset_path):
            logger.info(f"Dataset MGSM already downloaded")
            return

        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_path, 'de'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, 'fr'), exist_ok=True)

        logger.info("Downloading MGSM dataset")
        
        de = load_dataset("juletxara/mgsm", "de")
        fr = load_dataset("juletxara/mgsm", "fr")

        de.save_to_disk(os.path.join(dataset_path, 'de'))
        fr.save_to_disk(os.path.join(dataset_path, 'fr'))


        logger.info("Downloaded MGSM dataset!")

    def load(self):
        dataset_path = self._config.get_dataset_path(self.config_name())
        de_samples = load_from_disk(os.path.join(dataset_path, 'de'))
        fr_samples = load_from_disk(os.path.join(dataset_path, 'fr'))

        train_de = de_samples['train'].to_pandas()
        train_de["language"] = "de"
        self._train_samples_de = train_de.to_dict(orient="records")

        train_fr = fr_samples['train'].to_pandas()
        train_fr["language"] = "fr"
        self._train_samples_fr = train_fr.to_dict(orient="records")

        test_de = de_samples['test'].to_pandas()
        test_de["language"] = "de"

        test_fr = fr_samples['test'].to_pandas()
        test_fr["language"] = "fr"

        test_dataframes = pd.concat([test_de, test_fr])

        self._test_samples = test_dataframes.to_dict(orient="records")

    def get_test_samples(self) -> List:
        return self._test_samples

    def get_train_samples(self, language:str) -> List:
        if language == "de":
            return self._train_samples_de
        elif language == "fr":
            return self._train_samples_fr
        else:
            raise ValueError("Invalid language")
