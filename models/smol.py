from typing import List

import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils.custom_stopping import EosListStoppingCriteria
from our_datasets.base_dataset import EvalSample, Message
from models.base_model import BaseModel
import torch


class SmolModel(BaseModel):
    def __init__(
        self, model_name: str = "HuggingFaceTB/SmolLM-1.7B-Instruct", revision="v0.1"
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision
        )

    def generate(self, messages: List[Message]) -> Message:
        ## prepare chat history

        messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]

        ## apply chat template
        new_query = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        stop = [
            EosListStoppingCriteria(self._tokenizer.encode(self._tokenizer.eos_token))
        ]

        ## generate response
        inputs = self._tokenizer(
            new_query,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=True,
            padding=True,
        ).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                pad_token_id=self._tokenizer.eos_token_id,
                stopping_criteria=stop,
                max_new_tokens=100
            )
        out = out[
            :, inputs["input_ids"].shape[1] :
        ]  ## Keeping only newly generated tokens
        response = self._tokenizer.batch_decode(out, skip_special_tokens=True)

        return Message(role="assistant", content=response[0])
