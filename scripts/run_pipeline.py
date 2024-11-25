import argparse
import random

from datasets.base_dataset import BaseDataset, Message
from datasets.gsm8k_dataset import GSM8K
from datasets.math_dataset import MATH
from evalutaion.base_eval_task import EvalTask
from evalutaion.gsm8k_task import GSM8KNShot
from evalutaion.math_task import MATHFewShot
from models.ethel import EthelModel
from models.ollama import OllamaModel
from utils.config import Config
from utils.recorder import Recorder

if __name__ == '__main__':
    random.seed(239)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Dataset Evaluation")
    parser.add_argument("--dataset", required=True, help="The dataset to use for evaluation: GSM8K, MATH")
    parser.add_argument("--model", required=True, help="The model to use for evaluation: Ethel, Ollama")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of iterations")

    args = parser.parse_args()

    config = Config(
        config_path='config.yaml',
        dataset_dir='data'
    )

    # Choose dataset class to work with
    datasets = {
        'GSM8K': GSM8K,
        'MATH': MATH
    }

    try:
        dataset_class = datasets[args.dataset]
    except KeyError:
        raise ValueError(f"Invalid dataset: {args.dataset}. Supported datasets: {list(datasets.keys())}")

    # Define dataset instance and download if necessary
    dataset: BaseDataset = dataset_class(config)
    dataset.download()

    # Load data to the RAM
    dataset.load()

    # Run Dataset Evaluation
    eval_task_class = {
        'GSM8K': GSM8KNShot,
        'MATH': MATHFewShot
    }

    eval_task: EvalTask = eval_task_class[args.dataset](dataset)

    models = {
        'Ethel': EthelModel,
        'Ollama': OllamaModel
    }

    try:
        model_class = models[args.model]
    except KeyError:
        raise ValueError(f"Invalid model: {args.model}. Supported models: {list(models.keys())}")

    model = model_class()

    recorder = Recorder(config.get_records_path())

    i = 0
    for ex in eval_task:
        resp = model.generate(ex.messages)
        generated_answer = eval_task.extract_answer(resp.content)
        is_correct = eval_task.is_correct(ex, generated_answer)

        # Record the iteration data
        recorder.record({
            "input": [m.to_dict() for m in ex.messages],
            "target_answer": ex.target,
            "response": resp.content,
            "generated_answer": generated_answer,
            "is_correct": is_correct,
        })

        i += 1
        if args.limit is not None and i >= args.limit:
            break

    recorder.save('evaluation_records.json')
