import json
import os
from typing import List, Optional

import openai

from our_datasets.base_dataset import Message
from models.base_model import BaseModel


class EthelModel(BaseModel):
    def __init__(self, model_name: Optional[str]):
        self._model_name = model_name if model_name is not None else "swissai/ethel-70b-pretrain"
        self._api_key = self._get_api_key()

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
        messages = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in messages
        ]

        client = openai.Client(
            base_url="https://fmapi.swissai.cscs.ch",
            api_key=self._api_key,
        )

        response = client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            stream=False,
            timeout=120
        )
        content = response.choices[0].message.content
        return Message(role="assistant", content=content)
