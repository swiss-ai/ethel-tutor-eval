from abc import ABC, abstractmethod
from typing import Iterator
import os
import sys
from our_datasets.base_dataset import BaseDataset, EvalSample


### An eval task handles the iteration over the evaluation samples
### and also provides a method to extract the 'correct' answer from the sample
class EvalTask(ABC):
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    @abstractmethod
    def __iter__(self) -> Iterator[EvalSample]:
        pass

    @classmethod
    def extract_answer(cls, answer: str, language: str = "") -> str:
        pass

    def __len__(self):
        return len(self.dataset.get_test_samples())


class NShotTask(EvalTask, ABC):
    def __init__(self, dataset: BaseDataset, n: int) -> None:
        super().__init__(dataset)
        self.n = n
