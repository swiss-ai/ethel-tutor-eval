import json
import os
from typing import Dict, Any


class Recorder:
    def __init__(self, dataset_name: str, eval_task_name: str, model_name: str, output_dir: str):
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.eval_task_name = eval_task_name
        self.model_name = model_name
        os.makedirs(output_dir, exist_ok=True)
        self.records = []

    def record(self, data: Dict[str, Any]):
        self.records.append(data)

    def save(self, filename: str):
        output_path = os.path.join(self.output_dir, self.dataset_name, self.eval_task_name, self.model_name, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(
            f"Saving records to {output_path}"
        )

        with open(output_path, 'w') as f:
            json.dump(self.records, f, indent=4)