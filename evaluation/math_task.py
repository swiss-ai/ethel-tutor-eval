import re
import signal
import logging

from typing import Iterator, Optional, List

from our_datasets.base_dataset import EvalSample, BaseDataset, Message
from evaluation.base_eval_task import EvalTask

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


class MATHFewShot(EvalTask):
    def __init__(self, dataset: BaseDataset):
        super().__init__(dataset)

        if dataset.config_name() != "math":
            raise ValueError("Can't run MATH N-shot evaluation on non-MATH dataset")

    def __iter__(self) -> Iterator[EvalSample]:
        for ex in self.dataset.get_test_samples():
            n_shot = self._generate_n_shot_messages(ex["question"])
            description = f"{ex['level']}|{ex['category']}"
            yield EvalSample(
                messages=n_shot, target=ex["answer"], description=description
            )

    def _generate_n_shot_messages(self, question: str) -> List[Message]:
        n_shot_messages = []

        for ex in self._few_shot_samples():
            question_content = f"Problem:\n{ex['question']}"
            n_shot_messages.append(Message(role="user", content=question_content))

            answer_content = f"Solution:\n{ex['answer']}"
            n_shot_messages.append(Message(role="assistant", content=answer_content))

        question_message = Message(role="user", content="Problem:\n" + question)

        return n_shot_messages + [question_message]

    @staticmethod
    def _few_shot_samples() -> List[dict]:
        return [
            {
                "question": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
                "answer": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
            },
            {
                "question": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
                "answer": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nFinal Answer: The final answer is $24$. I hope it is correct.",
            },
            {
                "question": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
                "answer": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nFinal Answer: The final answer is $16$. I hope it is correct.",
            },
            {
                "question": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
                "answer": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
            },
        ]

    @classmethod
    def extract_answer(cls, answer: str) -> str:
        return normalize_final_answer(extract_final_answer(answer))

    def is_correct(self, sample: EvalSample, answer: str) -> bool:
        sample_answer = normalize_final_answer(
            remove_boxed(last_boxed_only_string(sample.target))
        )
        generated_answer = self.extract_answer(answer)

        return is_equiv(generated_answer, sample_answer)


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] != left:
            raise ValueError(f"Expected {left} but got {s[: len(left)]}")
        return s[len(left) :]

    left = "\\boxed{"

    if s[: len(left)] != left:
        raise ValueError(f"Expected {left} but got {s[: len(left)]}")

    if s[-1] != "}":
        raise ValueError("Expected }" + f" but got {s[-1]}")

    return s[len(left) : -1]


class SignalTimeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with SignalTimeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                logger.debug(f"Couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                logger.debug(f"Couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        logger.error(e)
        raise
    except Exception as e:
        logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def extract_final_answer(text: str) -> str:
    match = re.search(
        r"Final Answer: The final answer is(.*?)[.]",
        text,
    )
    if match:
        return match.group(1).strip()

    return "[invalidanswer]"


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer
