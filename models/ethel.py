import json
import os
from typing import List

import openai

from our_datasets.base_dataset import Message
from models.base_model import BaseModel


class EthelModel(BaseModel):
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-90B-Vision-Instruct", constrained: bool = False):
        self._model_name = model_name
        self._api_key = self._get_api_key()
        self._constrained = constrained

    @staticmethod
    def _get_api_key():
        api_key = os.getenv('SWISSAI_API').strip() ## On windows very important!
        if not api_key:
            raise ValueError(
                """
                Error: SWISSAI_API environment variable not set
                Please set your API on Linux by running:
                export SWISSAI_API='your-api-key'
                Or on Windows by running:
                set SWISSAI_API = 'your-api-key'
                You can find your API key at: http://148.187.108.173:8080/
                """
            )
        return api_key

    def generate(self, messages: List[Message]) -> Message:
        json_schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[\\w]+$"},
                    "population": {"type": "integer"},
                },
                "required": ["name", "population"],
            }
        )
        messages = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in messages
        ]
        print(messages)
        with open("records/message.json", "w") as f:
            json.dump(messages, f)
        client = openai.Client(api_key=self._api_key, base_url="https://fmapi.swissai.cscs.ch")
        if self._constrained:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "foo", "schema": json.loads(json_schema)},
            }
        else:
            response_format = None
        response = client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            response_format=response_format,
            stream=False,
        )
        content = response.choices[0].message.content
        return Message(role="assistant", content=content)
