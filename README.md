Proposed structure:

llm_evaluation_pipeline/
├── data/                    # Dataset handling
│   ├── gsm8k/
│   ├── math/
│   └── tutoreval/
├── models/                  # LLM interaction layer
│   ├── base_model.py        # Abstract base class for LLMs
│   ├── llama_model.py       # Llama-specific implementation
│   └── ethel_model.py       # Ethel-specific implementation
├── evaluation/              # Evaluation pipeline
│   ├── metrics.py           # Custom metrics implementation
│   ├── evaluator.py         # Core evaluation logic
│   └── utils.py             # Helper functions
├── notebooks/               # For exploratory data analysis
│   └── data_preparation.ipynb
├── tests/                   # Unit and integration tests
│   ├── test_data_loaders.py
│   ├── test_model_inference.py
│   └── test_evaluation.py
├── scripts/                 # Command-line tools
│   ├── run_pipeline.py      # Run entire evaluation pipeline
│   └── download_datasets.py # Helper for fetching datasets
├── requirements.txt         # Python dependencies
├── config.yaml              # Configurations (paths, hyperparams, etc.)
└── README.md                # Project documentation
