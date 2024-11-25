import random
import re
from typing import Iterator, List

from datasets.base_dataset import BaseDataset, Message, EvalSample
from evalutaion.base_eval_task import EvalTask


class GSM8KNShot(EvalTask):
    ANS_RE = re.compile(r"#### (-?[0-9.,]+)")
    INVALID_ANS = "[invalid]"

    def __init__(self, dataset: BaseDataset, n: int = 8):
        super().__init__(dataset)

        if dataset.config_name() != 'gsm8k':
            raise ValueError("Can't run GSM8K N-shot evaluation on non-GSM8K dataset")

        self.n = n
        # store n train samples to keep the same prompt for all test samples in this task
        self._n_shot_samples = []

    def __iter__(self) -> Iterator[EvalSample]:
        for ex in self.dataset.get_test_samples():
            n_shot = self._generate_n_shot_messages(ex['question'])
            yield EvalSample(messages=n_shot, target=ex['answer'])

    @staticmethod
    def _question_prompt(question: str):
        instruction = """
            Let's think step by step. At the end, you MUST write the answer as an integer after '####'.
           """

        return f"Question: {question} {instruction}"

    def _generate_n_shot_messages(self, question: str) -> List[Message]:
        if len(self._n_shot_samples) == 0:
            self._n_shot_samples = random.sample(self.dataset.get_train_samples(), self.n)

        n_shot_messages = []
        for sample in self._n_shot_samples:
            question_content = f"Question: {sample['question']}"
            n_shot_messages.append(Message(role="user", content=question_content))

            answer_content = f"Answer: {sample['answer']}"
            n_shot_messages.append(Message(role="assistant", content=answer_content))

        question_message = Message(role="user", content=self._question_prompt(question))
        return n_shot_messages + [question_message]

    @classmethod
    def extract_answer(cls, answer: str) -> str:
        match = cls.ANS_RE.search(answer)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return cls.INVALID_ANS
