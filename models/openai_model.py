from typing import List
import os
from openai import OpenAI
from our_datasets.base_dataset import Message
from models.base_model import BaseModel


class OpenAIModel(BaseModel):
    def __init__(self, model_name: str):
        if model_name is None:
            raise ValueError("OpenAIModel must have a model_name argument provided [e.g., gpt-3.5-turbo, gpt-4]")

        self._model = model_name

        # Get the API key from the environment variable
        self._api_key = os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise EnvironmentError("The OPENAI_API_KEY environment variable is not set.")

        # Initialize the OpenAI client with the API key
        self._client = OpenAI(api_key=self._api_key)

    def generate(self, messages: List[Message]) -> Message:
        try:
            # Convert the messages to the OpenAI API format
            openai_messages = [
                {"role": message.role, "content": message.content} for message in messages
            ]

            # Call the OpenAI API to generate a response
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=openai_messages
            )

            # Extract the content of the first choice
            content = completion.choices[0].message.content

            # Return the response content as a Message object
            return Message(
                role="assistant",
                content=content
            )

        except Exception as e:
            # Handle errors during the API call
            raise RuntimeError(f"OpenAI API request failed: {e}")