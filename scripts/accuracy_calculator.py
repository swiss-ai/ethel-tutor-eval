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
from our_datasets.base_dataset import BaseDataset, Message
from our_datasets.gsm8k_dataset import GSM8K
from our_datasets.math_dataset import MATH
from our_datasets.mgsm_dataset import MGSM_DE, MGSM_FR
from evaluation.base_eval_task import EvalTask, NShotTask
from evaluation.gsm8k_task import GSM8KNShot
from evaluation.mgsm_task import MGSMNShot
from evaluation.math_task import MATHFewShot
from models.ethel import EthelModel
from models.ollama import OllamaModel
from models.smol import SmolModel
from utils.config import Config
from utils.recorder import Recorder
import json

if __name__ == '__main__':
    random.seed(239)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Dataset Evaluation")
    parser.add_argument("--dataset", required=True, help="The dataset to use for evaluation: GSM8K, MATH, MGSM_DE or MGSM_FR")
    parser.add_argument("--model", required=True, help="The model to use for evaluation: Ethel, Ollama")
    parser.add_argument("--model_name", required=False, help="The model name to use for model API")
    parser.add_argument("--n_shot", type=int, default=8, help="Number of n-shot samples")


    args = parser.parse_args()

    json_path =f"{args.model}/{args.model_name}/{args.n_shot}" if args.model_name else f"{args.model}"
    if not os.path.exists(f"records/{args.dataset}/{json_path}.json"):
        raise ValueError(f"Record file not found, please run the script to create the results")
    
    with open(f"records/{args.dataset}/{json_path}.json", "r") as f:
        data = json.load(f)
    
    if len(data) == 0:
        raise ValueError(f"Record file is empty, please run the script to create the results")
    
    total_items = len(data)
    is_correct_items = sum(1 for item in data if item.get('is_correct') == "true")

    if total_items == 0:
        raise ValueError(f"No items found in the record file")
    
    print(f"Evaluated model {args.model} on dataset {args.dataset} with {total_items} examples:")
    print(f"Accuracy: {is_correct_items / total_items}")