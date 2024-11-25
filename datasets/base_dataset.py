from abc import ABC, abstractmethod

from utils.config import Config


class BaseDataset(ABC):
    def __init__(self, config: Config):
        self._config = config

    @abstractmethod
    def _config_name(self) -> str:
        """Name of dataset as used in the config"""

    @abstractmethod
    def download(self):
        """Download dataset and saves locally"""
        pass

    @abstractmethod
    def prompt(self) -> str:
        """Return prompt for dataset"""
        pass

    @abstractmethod
    def __iter__(self):
        """Iterate over dataset"""
        pass
