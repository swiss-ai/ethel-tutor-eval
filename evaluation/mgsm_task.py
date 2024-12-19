import random
import re
from typing import Iterator, List
import sys
import os
from evaluation.base_eval_task import EvalTask, NShotTask
from our_datasets.base_dataset import BaseDataset, Message, EvalSample


class MGSMNShot(NShotTask):
    ANS_RE = re.compile(r"#### (-?[0-9.,]+)")
    INVALID_ANS = "[invalid]"

    def __init__(self, dataset: BaseDataset, n: int = 8):
        super().__init__(dataset, n)

        if dataset.config_name() != "mgsm":
            raise ValueError("Can't run MGSM N-shot evaluation on non-MGSM dataset")

        self.n = n
        # store n train samples to keep the same prompt for all test samples in this task
        self._n_shot_samples = []

    def __iter__(self) -> Iterator[EvalSample]:
        for ex in self.dataset.get_test_samples():
            n_shot = self._generate_n_shot_messages(ex["question"], ex["language"])
            yield EvalSample(messages=n_shot, target=str(ex["answer_number"]))

    @staticmethod
    def _question_prompt(question: str, language: str) -> str:
        if language == "de":
            instruction = "Denken Sie Schritt für Schritt. Am Ende schreiben Sie die Antwort als Ganzzahl nach dem: ####"
            return f"Frage: {question} {instruction} "
        elif language == "fr":
            instruction = "Pensez étape par étape. À la fin, vous DEVEZ écrire la réponse en tant qu'entier après ####"
            return f"Question : {question} {instruction} "
        else:
            raise ValueError("Invalid language")

    def _generate_n_shot_messages(self, question: str, language: str) -> List[Message]:
        if len(self._n_shot_samples) == 0:
            self._n_shot_samples = random.sample(
                self.dataset.get_train_samples(), self.n
            )

        n_shot_messages = []
        for sample in self._n_shot_samples:
            if language == "de":
                question_content = f"{sample['question']}"
                n_shot_messages.append(Message(role="user", content=question_content))

                answer_content = f"{sample['answer']} Die Antwort lautet #### {sample['answer_number']} "
                n_shot_messages.append(
                    Message(role="assistant", content=answer_content)
                )
            elif language == "fr":
                question_content = f"{sample['question']}"
                n_shot_messages.append(Message(role="user", content=question_content))

                answer_content = (
                    f"{sample['answer']} La réponse est #### {sample['answer_number']} "
                )
                n_shot_messages.append(
                    Message(role="assistant", content=answer_content)
                )
            else:
                raise ValueError("Invalid language")
        question_message = Message(
            role="user", content=self._question_prompt(question, language)
        )
        return n_shot_messages + [question_message]

    def is_correct(self, sample: EvalSample, answer: str) -> bool:
        # print(self.extract_answer(sample.target), answer)
        return self.extract_answer(sample.target) == answer

    @classmethod
    def extract_answer(cls, answer: str) -> str:
        # print("answer", answer)
        match = cls.ANS_RE.search(answer)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        parts = answer.split("Die Antwort")
        if len(parts) == 1:  ## TODO! Language check, not like this
            parts = answer.split("La réponse")
        last_part = parts[-1].strip()

        # Step 3: Extract the number using regex
        match = re.search(r"-?\d+(\.\d+)?", last_part)
        if match:
            match_str = match.group()
            return match_str
        else:
            return cls.INVALID_ANS
