from abc import ABC, abstractmethod
from typing import List
from datasets.base_dataset import Message


class BaseModel(ABC):
    """Abstract base class for LLMs."""

    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        """Generate a response for a given prompt."""
        pass