import argparse
import random
import tqdm
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)

## if you are in scripts folder
os.chdir(parent_dir)

from utils.config import Config
from evaluation.gsm8k_task import GSM8KNShot
from evaluation.base_eval_task import EvalTask
from evaluation.math_task import MATHFewShot
from evaluation.tutoreval_task import TutorEvalTask
from our_datasets.gsm8k_dataset import GSM8K
#from datasets.gsm8k_dataset import GSM8K
from our_datasets.base_dataset import BaseDataset
from our_datasets.math_dataset import MATH
from our_datasets.tutoreval_dataset import TutorEval
from models.ethel import EthelModel
from models.ollama import OllamaModel
from models.smol import SmolModel
#from grader_models.smol_grader import SmolGrader

from utils.recorder import Recorder

if __name__ == '__main__':
    random.seed(239)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Dataset Evaluation")
    parser.add_argument("--dataset", required=True, help="The dataset to use for evaluation: TutorEval")
    parser.add_argument("--model", required=True, help="The model to use for evaluation: Ethel, Ollama, Smol")
    parser.add_argument("--grader_model", required=True, help="The model to use for grading: Ethel, Ollama, Smol")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of iterations")
    parser.add_argument("--closed_book", required=False, default=True, help="True, if closed_book evaluation is desired, False if the open_book evaluation is desired")
    #parser.add_argument("--difficulty", required=False, help="The difficulty level of the question")
    args = parser.parse_args()

    config = Config(
        config_path='/home/smaldo/Desktop/machine_learning_CS-433/ethel-tutor-eval/config.yaml',
        dataset_dir='data'
    )

    # Choose dataset class to work with
    datasets = {
        'GSM8K': GSM8K,
        'MATH': MATH,
        'TutorEval': TutorEval
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

    if args.dataset == 'TutorEval':
        dataset.choose_book(args.closed_book)
    # Run Dataset Evaluation
    eval_task_class = {
        'GSM8K': GSM8KNShot,
        'MATH': MATHFewShot,
        'TutorEval': TutorEvalTask
    }

    eval_task: EvalTask = eval_task_class[args.dataset](dataset)
    models = {
        'Ethel': EthelModel,
        'Llama': EthelModel,
        'Ollama': OllamaModel,
        'Smol': SmolModel,
    }


    model_names = {
        "Ethel": "swissai/ethel-70b-tutorchat",
        "Llama": "swissai/ethel-70b-magpie",
        "Ollama": "llama3.2",
        "Smol": "HuggingFaceTB/SmolLM-1.7B-Instruct"
    }

    try:
        model_class = models[args.model]
    except KeyError:
        raise ValueError(f"Invalid model: {args.model}. Supported models: {list(models.keys())}")

    try:
        grader_model_class = models[args.grader_model]
    except KeyError:
        raise ValueError(f"Invalid grader model: {args.grader_model}. Supported models: {list(models.keys())}")

    model = model_class(model_name = model_names[args.model])
    #grader_model = grader_model_class()
    grader_model = grader_model_class(model_name=model_names[args.grader_model])  
    
    #    recorder = Recorder(config.get_records_path())
    eval_task_name = args.dataset
    model_name = args.model
    output_dir = config.get_records_path()
    recorder = Recorder("dataset","ss", model_name, output_dir)

    i = 0
    all_grades = []
    for ex in tqdm.tqdm(eval_task, total=len(eval_task)):
        tutor_response = model.generate(ex.messages)
        grader_response, grades = eval_task.grade(ex, tutor_response, grader_model)

        # Record the iteration data
        recorder.record({
            "input": [m.to_dict() for m in ex.messages],
            "key_points": ex.target,
            "tutor_response": tutor_response.content,
            "grader_response": grader_response,
            "presentation": grades[0],
            "correctness": grades[1],
            "difficulty": ex.difficulty,
            "domain" : ex.domain
        })
        all_grades.append((grades[0], grades[1]))

        i += 1
        if args.limit is not None and i >= args.limit:
            break

    print(f"Evaluated model {args.model} on dataset {args.dataset} with {i} examples and {args.grader_model} grader model:")
    print(f"Presentation: {sum([grade[0] for grade in all_grades])/len(all_grades)}")
    print(f"Correctness: {sum([grade[1] for grade in all_grades])/len(all_grades)}")

    recorder.save('evaluation_records_tutor.json')
