import argparse
import os.path

from datasets.base_dataset import BaseDataset
from datasets.gsm8k_dataset import GSM8K
from datasets.math_dataset import MATH
from utils.config import Config

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Dataset Evaluation")
    parser.add_argument("--dataset", required=True, help="The dataset to use for evaluation: GSM8K, MATH")
    parser.add_argument("--model", required=True, help="The model to use for evaluation: Ethel, Ollama")

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

    # Run Dataset Evaluation
