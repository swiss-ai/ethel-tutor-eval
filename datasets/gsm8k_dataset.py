import io
import os
import random
import re
import logging
import shutil
import zipfile
from typing import List, Iterator

import requests

from datasets.base_dataset import BaseDataset, Message
from utils.config import Config
from utils.read_utils import read_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GSM8K(BaseDataset):
    def __init__(self, config: Config):
        super().__init__(config)
        self._train_samples = []
        self._test_samples = []

        # store idx shuffle, unique for the instance, to use as n-shot learning examples
        self._train_idx_shuffle = []

    def _config_name(self) -> str:
        return "gsm8k"

    def download(self):
        dataset_path = self._config.get_dataset_path(self._config_name())

        if os.path.exists(dataset_path):
            logger.info(f"Dataset GSM8k already downloaded")
            return

        os.makedirs(dataset_path, exist_ok=True)

        logger.info("Downloading GSM8k dataset")
        dataset_url = 'https://github.com/openai/grade-school-math/archive/refs/heads/master.zip'

        response = requests.get(dataset_url)
        response.raise_for_status()

        # Extracting the zip file content
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            for member in zip_ref.namelist():
                # We need to extract only files from the data directory
                if member.startswith(os.path.join('grade-school-math-master', 'grade_school_math', 'data') + os.sep):
                    if '.' not in member.split(os.sep)[-1]:
                        continue  # ignore /data/ directory
                    # This will extract the file into the desired directory structure
                    zip_ref.extract(member, dataset_path)
                    # Rename the extracted files to remove the prefix directories
                    extracted_path = os.path.join(dataset_path, member)
                    target_path = os.path.join(dataset_path, os.path.basename(member))
                    shutil.move(extracted_path, target_path)

        extracted_main_dir = os.path.join(dataset_path, 'grade-school-math-master')
        if os.path.exists(extracted_main_dir):
            shutil.rmtree(extracted_main_dir)

        logger.info("Downloaded GSM8k dataset!")

    def load(self):
        dataset_path = self._config.get_dataset_path(self._config_name())
        self._test_samples = read_jsonl(os.path.join(dataset_path, 'test.jsonl'))
        self._train_samples = read_jsonl(os.path.join(dataset_path, 'train.jsonl'))

    @staticmethod
    def _question_prompt(question: str):
        instruction = """
         Let's think step by step. At the end, you MUST write the answer as an integer after '####'.
        """

        return f"Question: {question} {instruction}"

    def _generate_n_shot_messages(self, question: str, n: int = 8) -> List[Message]:
        if len(self._train_idx_shuffle) == 0:
            train_idx = list(range(len(self._train_samples)))
            random.shuffle(train_idx)
            self._train_idx_shuffle = train_idx

        n = min(n, len(self._train_samples))

        n_shot_samples = [self._train_samples[i] for i in self._train_idx_shuffle[:n]]

        n_shot_messages = []

        for sample in n_shot_samples:
            question_content = f"Question: {sample['question']}"
            n_shot_messages.append(Message(role="user", content=question_content))

            answer_content = f"Answer: {sample['answer']}"
            n_shot_messages.append(Message(role="assistant", content=answer_content))

        question_message = Message(role="user", content=self._question_prompt(question))
        return n_shot_messages + [question_message]

    def __iter__(self) -> Iterator[List[Message]]:
        for ex in self._test_samples:
            n_shot = self._generate_n_shot_messages(ex['question'], n=8)
            yield n_shot



def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer
