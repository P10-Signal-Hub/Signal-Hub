import os

import torch
from huggingface_hub import login
from transformers import AutoTokenizer, pipeline

class Model:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if (cls._instance is None):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:

            login(token=os.environ['HUGGINGFACE_TOKEN'])

            self.model_name = 'google/gemma-3-4b-it'
            self.model_task = 'text-generation'

            # Use GPU if available, otherwise use CPU
            device = 0 if torch.cuda.is_available() else -1
            self.device_type = "GPU" if device == 0 else "CPU"
            # Create tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            # Sets what device to use for the pipeline. 0 = GPU, -1 = CPU
            # Creation of the pipeline. Task is the AI task to perform (Example: 'text-generation'). Model is the name of the model. dtype is the data type to use, this is set to auto to let the pipeline determine the best data type to use.
            self.model = pipeline(task=self.model_task,
                                    model=self.model_name,
                                    tokenizer=tokenizer,
                                    device=device,
                                    dtype='auto')

            print(f"Model [{self.model_name}] initialized on {self.device_type}")

    def get_model_information(self):
        return {"model_name": self.model_name, "model_task": self.model_task, "device_type": self.device_type}