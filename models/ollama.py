from typing import List

import requests

from our_datasets.base_dataset import Message
from models.base_model import BaseModel


class OllamaModel(BaseModel):
    def __init__(self,  model_name: str, ollama_api: str = "http://localhost:11434/api/chat"):
        self._ollama_api = ollama_api
        if model_name is None:
            raise ValueError("OllamaModel must have a model_name argument provided [llama3.2|smollm]")
        self._model = model_name

    def generate(self, messages: List[Message]) -> Message:
        messages = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in messages
        ]

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False
            }
        )

        response_data = response.json()
        response.raise_for_status()

        content = response_data['message']['content']

        return Message(
            role="assistant",
            content=content
        )
