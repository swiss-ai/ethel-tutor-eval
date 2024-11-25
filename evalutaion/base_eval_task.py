from abc import ABC
from typing import Iterator

from datasets.base_dataset import BaseDataset, EvalSample


class EvalTask(ABC):
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    def __iter__(self) -> Iterator[EvalSample]:
        pass

    @classmethod
    def extract_answer(cls, answer: str) -> str:
        return answer

    def is_correct(self, sample: EvalSample, answer: str) -> bool:
        return answer == sample.target