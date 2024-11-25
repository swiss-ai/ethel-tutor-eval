import json
import os
from typing import Dict, Any


class Recorder:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.records = []

    def record(self, data: Dict[str, Any]):
        self.records.append(data)

    def save(self, filename: str):
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            json.dump(self.records, f, indent=4)