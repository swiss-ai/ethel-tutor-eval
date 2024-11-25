import io
import os
import re
import logging
import shutil
import zipfile

import requests

from datasets.base_dataset import BaseDataset
from utils.read_utils import read_jsonl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GSM8K(BaseDataset):
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

    def prompt(self):
        pass

    def __iter__(self):
        pass


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
