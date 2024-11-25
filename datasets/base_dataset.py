import dataclasses
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
    def load(self):
        """Load dataset from local storage"""
        pass

    @abstractmethod
    def __iter__(self):
        """Iterate over dataset"""
        pass

@dataclasses.dataclass
class Message:
    role: str
    content: str
