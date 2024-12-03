# Ethel Tutor Eval

This repository contains small framework for evaluating Ethel on education-specific benchmarks.


## User guide

Use the following command with the `dataset` and `model` command line arguments to run the evaluation

```bash

python3 -m scripts.run_pipeline --dataset=GSM8K --model=Ollama
```

## Datasets

Currently, our evaluation pipeline supports the following datasets:

- [GSM8K](https://github.com/openai/grade-school-math)
- [MATH](https://github.com/hendrycks/math)


### Authors:

This work was done as part of the ML4Science project at CS-433 course at EPFL.

The main contributors are:
- Kamel Charaf
- Ivan Pavlov
- Michele Smaldone

