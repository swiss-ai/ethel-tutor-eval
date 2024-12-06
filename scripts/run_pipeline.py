import argparse
import logging
import os
import random
import tqdm
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

## if you are in scripts folder
# os.chdir(parent_dir)
from our_datasets.base_dataset import BaseDataset, Message
from our_datasets.gsm8k_dataset import GSM8K
from our_datasets.math_dataset import MATH
from our_datasets.mgsm_dataset import MGSM
from evaluation.base_eval_task import EvalTask
from evaluation.gsm8k_task import GSM8KNShot
from evaluation.mgsm_task import MGSMNShot
from evaluation.math_task import MATHFewShot
from models.ethel import EthelModel
from models.ollama import OllamaModel
from models.smol import SmolModel
from utils.config import Config
from utils.recorder import Recorder


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    random.seed(239)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Dataset Evaluation")
    parser.add_argument("--dataset", required=True, help="The dataset to use for evaluation: GSM8K, MATH, MGSM")
    parser.add_argument("--model", required=True, help="The model to use for evaluation: Ethel, Ollama")
    parser.add_argument("--model_name", required=False, help="The model name to use for model API")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of iterations")
    parser.add_argument("--n_shot", type=int, default=8, help="Number of n-shot samples")

    args = parser.parse_args()

    config = Config(
        config_path='config.yaml',
        dataset_dir='data'
    )

    # Choose dataset class to work with
    datasets = {
        'GSM8K': GSM8K,
        'MATH': MATH,
        'MGSM': MGSM,
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
        'MATH': MATHFewShot,
        'MGSM': MGSMNShot,
    }

    eval_task: EvalTask = eval_task_class[args.dataset](dataset, args.n_shot)

    models = {
        'Ethel': EthelModel,
        'Ollama': OllamaModel,
        'Smol': SmolModel,
    }

    model_args_dict = {
        'Ethel': {
            'model_name': args.model_name
        },
        'Ollama': {

        }
    }

    try:
        model_class = models[args.model]
        model_args = model_args_dict[args.model]
    except KeyError:
        raise ValueError(f"Invalid model: {args.model}. Supported models: {list(models.keys())}")

    model = model_class(**model_args)

    recorder = Recorder(
        output_dir=config.get_records_path(),
        dataset_name=args.dataset,
        eval_task_name=eval_task.__class__.__name__,
        model_name=f"{args.model}/{args.model_name}" if args.model_name else f"{args.model}"
    )

    i = 0

    is_correct_labels = []
    failed_samples = []

    total_len = len(eval_task)
    if args.limit is not None:
        total_len = min(args.limit, total_len)

    for ex in tqdm.tqdm(eval_task, total=total_len):
        try:
            resp = model.generate(ex.messages)
        except:
            logger.error(f"Failed to generate response for sample {i}")
            failed_samples.append({
                "sample_id": i,
                "input": [m.to_dict() for m in ex.messages],
                "target_answer": ex.target
            })
        generated_answer = eval_task.extract_answer(resp.content)
        is_correct = eval_task.is_correct(ex, generated_answer)

        is_correct_labels.append(is_correct)

        # Record the iteration data
        recorder.record({
            "input": [m.to_dict() for m in ex.messages],
            "target_answer": ex.target,
            "response": resp.content,
            "generated_answer": generated_answer,
            "is_correct": is_correct
        })

        if i % 100 == 0:
            recorder.save('evaluation_records.json')
            recorder.save_failed(failed_samples, 'failed_evaluation_records.json')

        i += 1
        if args.limit is not None and i >= args.limit:
            break

    print(f"Evaluated model {args.model} on dataset {args.dataset} with {len(is_correct_labels)} examples:")
    print(f"Accuracy: {sum(is_correct_labels) / len(is_correct_labels)}")

    recorder.save('evaluation_records.json')
    recorder.save_failed(failed_samples, 'failed_evaluation_records.json')
