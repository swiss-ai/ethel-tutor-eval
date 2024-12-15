# Ethel Tutor Eval

This repository contains a framework for evaluating Ethel on education-specific benchmark datasets.

## Datasets

Currently, our evaluation pipeline supports the following datasets:

- [GSM8K](https://github.com/openai/grade-school-math)
- [MATH](https://github.com/hendrycks/math)
- [MGSM](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mgsm)
- [TutorEval](https://github.com/princeton-nlp/LM-Science-Tutor)

## Scripts

### For using the GSM8K, MATH or MGSM dataset with the evaluation pipeline
Use the following command with the `dataset` and `model` command line arguments to run the evaluation

**Required arguments:**
- --`dataset`: The dataset to be used for the evaluation, e.g.: `GSM8K`
- --`model`: The model type to generate the answer, e.g.: `Ethel`

**Optional arguments:**
- -- `model_name`: The exact name of the model, e.g.: `swissai/ethel-70b-magpie`
- -- `limit`: If you prefer to run the evaluation on a subset of the dataset
- -- `n_shot`: If you want to perform n-shot learning

**Example:**
```bash
python3 -m scripts.run_pipeline --dataset=MATH --model=Ethel
```

### For running the TutorEval dataset with the evaluation pipeline

**Required arguments:**
- -- `dataset`: The dataset to be used for the evaluation, e.g.: `GSM8K`
- -- `model`: The model type that works as a tutor to generate the answer, e.g.: `Ethel`
- -- `grader_model`: The model type to grade the tutor's answer, e.g.: `Ethel`

**Optional arguments:**
- -- `model_name`: The exact name of the model, e.g.: swissai/ethel-70b-magpie
- -- `grader_model_name`: The exact name of the model, e.g.: swissai/ethel-70b-magpie
- -- `limit`: If you prefer to run the evaluation on a subset of the dataset
- -- `closed_book`: If you want to evaluate on a closed_book setup (or on open_book)

```bash

python3 -m scripts.run_tutoreval --dataset=TutorEval --model=Ethel --grader_model=Ethel
```

### Authors:

This work was done as part of the ML4Science project at the CS-433 course at EPFL.

The main contributors are:
- Kamel Charaf
- Ivan Pavlov
- Michele Smaldone

