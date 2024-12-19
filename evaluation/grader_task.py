import re
import signal
import logging

from typing import Iterator, Optional, List

from our_datasets.base_dataset import EvalSample, BaseDataset, Message
from evaluation.base_eval_task import EvalTask
from models.base_model import BaseModel

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`sympy` is required for generating translation task prompt templates. \
please install sympy via pip install lm-eval[math] or pip install -e .[math]",
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# The answer processing code inspired by https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
"""
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
"""


class GraderTask(EvalTask):

    ANS_RE = re.compile(r"####\s*(.*)")
    INVALID_ANS = []

    def __init__(self, dataset: BaseDataset, closed_book: bool = True):
        super().__init__(dataset)

        if dataset.config_name() != "Grader":
            raise ValueError("Can't run Grader N-shot evaluation on non-Grader dataset")

        if closed_book:
            with open("templates/closedbook_generation_template.txt", "r") as f:
                self._template = f.read()
            with open("templates/closedbook_grading_template.txt", "r") as f:
                self._grading_template = f.read()
        else:
            with open("templates/openbook_generation_template.txt", "r") as f:
                self._template = f.read()
            with open("templates/openbook_grading_template.txt", "r") as f:
                self._grading_template = f.read()

    def __iter__(self) -> Iterator[EvalSample]:
        for ex in self.dataset.get_test_samples():
            # query = self._template.replace("{{QUESTION}}", ex['question']).replace("{{CHAPTER}}", ex['chapter'])
            answer = ex.get("answer")
            all_key_points = ex.get("original_key_points")
            message = self.generate_grader_prompt(answer, all_key_points)
            target = ex.get("key_points_mask")
            target.remove("_")
            messages = [Message(role="user", content=message)]
            description = f"{ex.get('closed_book')}|{ex.get('answer_in_chapter')}|{ex.get('misleading_question')}"
            yield EvalSample(
                messages=messages,
                target=target,
                description=description,
                difficulty=ex.get("difficulty"),
                domain=ex.get("domain"),
            )

    @staticmethod
    def _question_prompt(question: str):
        instruction = """
            Your task is to decide which of the following .
           """
        return f"Question: {question} {instruction}"

    def generate_grader_prompt(self, answer: str, all_key_points: List[str]) -> str:
        instruction = (
            "Your task is to decide which of key points are present in the answer provided and return the letter of the key points that are present in the answer. Here are the key points:\n"
            + "\n".join([f"{chr(65+i)}. {kp}" for i, kp in enumerate(all_key_points)])
        )
        instruction += (
            "\nAnd then the answer is: \n"
            + answer
            + "\n\n Return the letters separated by commas after ####"
        )
        return instruction

    def _generate_n_shot_messages(self, question: str) -> List[Message]:
        pass

    @staticmethod
    def _few_shot_samples() -> List[dict]:
        return []

    def grade(
        self, init_query: EvalSample, generated_answer: str, grader_model: BaseModel
    ) -> str:
        prompt = (
            self._grading_template.replace(
                "{{QUESTION}}", init_query.messages[0].content
            )
            .replace("{{KEY_POINTS}}", init_query.target[0])
            .replace("{{OUTPUT}}", generated_answer.content)
        )
        grading_messages = [Message(role="user", content=prompt)]
        grader_response = grader_model.generate(grading_messages)
        return (grader_response.content, self.extract_answer(grader_response.content))

    @classmethod
    def extract_answer(cls, answer: str) -> str:
        match = cls.ANS_RE.search(answer)
        if match:
            match_str = match.group(1).split(",")
            elements = [element.strip() for element in match_str]
            return elements
        else:
            return cls.INVALID_ANS

    def calculate_edit_distance(self, sample: EvalSample, answer: str) -> bool:
        # Calculate the edit distance between the target and the generated answer
        return self.jaccard_distance(sample.target, answer)

    def jaccard_distance(
        self, guess_answer: List[str], true_answer: List[str]
    ) -> float:
        set_guess = set(guess_answer)
        set_true = set(true_answer)

        intersection = len(set_guess.intersection(set_true))
        union = len(set_guess.union(set_true))

        jaccard_index = intersection / union if union > 0 else 0

        return 1 - jaccard_index
