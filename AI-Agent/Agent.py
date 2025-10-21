import gc
import time

import torch

from Model import Model
class Agent:
    def __init__(self, model:Model):
        self.model = model

    # ---Handles the API request---
    def input_handler(self, payload, method):
        try:
            if payload is None:
                return {'error': 'No payload provided.'}

            print(f'Finding event for {payload["request_id"]}')
            print(f'Conversation: {payload["conversation"]}')

            start_time = time.time()

            information = self.model.get_model_information()
            # Force garbage collection and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Find important context for the prompts
            results = method(payload)
            end_time = time.time()
            information['elapsed_time'] = end_time - start_time
            output = {'result': results, 'model_information': information}
            return output
        except Exception as e:
            return {'error': str(e)}