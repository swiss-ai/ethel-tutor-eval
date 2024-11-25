from typing import List

import requests

from datasets.base_dataset import Message
from models.base_model import BaseModel


class OllamaModel(BaseModel):
    def __init__(self, ollama_api: str = "http://localhost:11434/api/chat", model_name: str = "llama3.2"):
        self._ollama_api = ollama_api
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
                "model": "llama3.2",
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
