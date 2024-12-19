from abc import ABC, abstractmethod
from typing import List
from our_datasets.base_dataset import EvalSample, Message


class BaseModel(ABC):
    """
    Abstract base class for LLMs.
    Contains a method to generate a response for a given list of Messages.
    """

    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        """Generate a response for a given prompt."""
        pass
