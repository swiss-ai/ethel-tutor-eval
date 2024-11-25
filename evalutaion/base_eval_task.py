from abc import ABC, abstractmethod
from typing import Iterator

from datasets.base_dataset import BaseDataset, EvalSample


class EvalTask(ABC):
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    @abstractmethod
    def __iter__(self) -> Iterator[EvalSample]:
        pass

    @classmethod
    def extract_answer(cls, answer: str) -> str:
        pass

    def is_correct(self, sample: EvalSample, answer: str) -> bool:
        return answer == sample.target

    def __len__(self):
        return len(self.dataset.get_test_samples())
