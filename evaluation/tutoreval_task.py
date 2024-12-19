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


# The grading processing code inspired by https://github.com/princeton-nlp/LM-Science-Tutor?tab=readme-ov-file


class TutorEvalTask(EvalTask):

    def __init__(self, dataset: BaseDataset, closed_book: bool = True):
        super().__init__(dataset)

        if dataset.config_name() != "TutorEval":
            raise ValueError(
                "Can't run TutorEval N-shot evaluation on non-TutorEval dataset"
            )

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
            query = self._template.replace("{{QUESTION}}", ex["question"]).replace(
                "{{CHAPTER}}", ex["chapter"]
            )
            messages = [Message(role="user", content=query)]
            description = f"{ex['closed_book']}|{ex['answer_in_chapter']}|{ex['misleading_question']}"
            yield EvalSample(
                messages=messages,
                target=ex["key_points"],
                description=description,
                difficulty=ex["difficulty"],
                domain=ex["domain"],
            )

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
    def extract_answer(cls, answer: str) -> str | List[float]:
        grades = [float(d) for d in re.findall(pattern=r":\s?(\d.*)/3", string=answer)]
        if len(grades) == 0:
            grades = [-1.0, -1.0]
        return grades
