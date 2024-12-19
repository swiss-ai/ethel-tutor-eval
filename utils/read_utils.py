import json
from typing import List


def read_jsonl(path: str) -> List:
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
