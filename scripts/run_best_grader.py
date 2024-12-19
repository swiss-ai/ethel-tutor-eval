import argparse
import random
import tqdm
import os
import sys

from models.openai_model import OpenAIModel

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

## if you are in scripts folder
# os.chdir(parent_dir)
# os.chdir(parent_dir)
import logging
from utils.config import Config
from evaluation.gsm8k_task import GSM8KNShot
from evaluation.base_eval_task import EvalTask
from evaluation.math_task import MATHFewShot
from evaluation.tutoreval_task import TutorEvalTask
from evaluation.grader_task import GraderTask
from our_datasets.gsm8k_dataset import GSM8K

# from datasets.gsm8k_dataset import GSM8K
from our_datasets.base_dataset import BaseDataset
from our_datasets.math_dataset import MATH
from our_datasets.grader_dataset import Grader
from our_datasets.tutoreval_dataset import TutorEval
from models.ethel import EthelModel
from models.ollama import OllamaModel
from models.smol import SmolModel

# from grader_models.smol_grader import SmolGrader

from utils.recorder import Recorder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    random.seed(239)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Dataset Evaluation")
    parser.add_argument(
        "--grader_model",
        required=True,
        help="The model to use for grading: Ethel, Ollama, Smol",
    )
    parser.add_argument(
        "--grader_model_name",
        required=False,
        help="The grader model name to use for model API",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit the number of iterations"
    )
    parser.add_argument(
        "--closed_book",
        required=False,
        default=True,
        help="True, if closed_book evaluation is desired, False if the open_book evaluation is desired",
    )

    # parser.add_argument("--difficulty", required=False, help="The difficulty level of the question")
    args = parser.parse_args()
    config = Config(config_path="config.yaml", dataset_dir="data")

    # Choose dataset class to work with
    datasets = {"Grader": Grader}
    dataset_name = "Grader"
    dataset_class = datasets[dataset_name]

    # Define dataset instance and download if necessary
    dataset: BaseDataset = dataset_class(config)
    dataset.download()

    # Load data to the RAM
    dataset.choose_book(args.closed_book)
    dataset.load()

    # Run Dataset Evaluation
    eval_task_class = {"Grader": GraderTask}
    eval_task: EvalTask = eval_task_class[dataset_name](dataset)
    models = {
        "Ethel": EthelModel,
        "Ollama": OllamaModel,
        "Smol": SmolModel,
        "OpenAI": OpenAIModel,
    }

    model_args_dict = {
        "Ethel": {"model_name": args.grader_model_name},
        "Ollama": {"model_name": args.grader_model_name},
        "OpenAI": {"model_name": args.grader_model_name},
    }

    try:
        model_class = models[args.grader_model]
        model_args = model_args_dict[args.grader_model]
    except KeyError:
        raise ValueError(
            f"Invalid model: {args.grader_model}. Supported models: {list(models.keys())}"
        )

    model = model_class(**model_args)

    recorder = Recorder(
        output_dir=config.get_records_path(),
        dataset_name=dataset_name,
        eval_task_name=eval_task.__class__.__name__,
        model_name=(
            f"grader/{args.grader_model}/{args.grader_model_name}"
            if args.grader_model_name
            else f"{args.grader_model}"
        ),
    )

    values = []
    failed_samples = []

    total_len = len(eval_task)
    if args.limit is not None:
        total_len = min(args.limit, total_len)
    i = 0
    for ex in tqdm.tqdm(eval_task, total=total_len):
        # print(ex.messages[0].content)
        try:
            resp = model.generate(ex.messages)
        except:
            logger.error(f"Failed to generate response for sample {i}")
            failed_samples.append(
                {
                    "sample_id": i,
                    "input": [m.to_dict() for m in ex.messages],
                    "target_answer": ex.target,
                }
            )
            i += 1
            if args.limit is not None and i >= args.limit:
                break
            continue
        generated_answer = eval_task.extract_answer(resp.content)
        # print("Generated Answer: ", generated_answer)
        edit_distance = eval_task.calculate_edit_distance(ex, generated_answer)

        values.append(edit_distance)

        # Record the iteration data
        #
        recorder.record(
            {
                "input": [m.to_dict() for m in ex.messages],
                "target_answer": ex.target,
                "response": resp.content,
                "generated_answer": generated_answer,
                "edit_distance": edit_distance,
                "description": ex.description,
                "index": i,
            }
        )

        if i % 100 == 0:
            recorder.save("evaluation_records.json")
            recorder.save_failed(failed_samples, "failed_evaluation_records.json")

        i += 1
        if args.limit is not None and i >= args.limit:
            break
    if len(values) == 0:
        raise ValueError(
            f"No items was evaluated, please run the script again to create the results"
        )
    print(
        f"Evaluated model {args.grader_model} on dataset {dataset_name} with {len(values)} examples:"
    )
    print(f"Accuracy: {sum(values) / len(values)}")

    recorder.save("evaluation_records.json")
    recorder.save_failed(failed_samples, "failed_evaluation_records.json")
