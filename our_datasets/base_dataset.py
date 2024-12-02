import dataclasses
from abc import ABC, abstractmethod
from typing import List
import sys
import os
from utils.config import Config


class BaseDataset(ABC):
    def __init__(self, config: Config):
        self._config = config

    @abstractmethod
    def config_name(self) -> str:
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
    def get_test_samples(self) -> List:
        """Get test samples for evaluation"""
        pass

    @abstractmethod
    def get_train_samples(self) -> List:
        """Get train samples for n-shot evaluating"""
        pass


@dataclasses.dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class EvalSample:
    messages: List[Message]
    target: str
    description: str = ""
