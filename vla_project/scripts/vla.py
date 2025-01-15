''' 
This script provides a class for VLA inference, the model itself is hosted on a remote server. 
We will be using Runpod to host the model. If you want to run it locally, strip the server calls from this script and link the model locally.
The server deploys the model (through HF AutoClass API) over REST API.

action_space = {
'normalized': {
    'action': {
        'mask': [True, False, True],
        'q01': [-1, -1, -1],
        'q99': [1, 1, 1]
            }
    }
}
'''
# client side specific
import requests
import json_numpy
json_numpy.patch()
import numpy as np
from vla_project.utils.util import AttributeDict
 
SERVER_LOC = "http://0.0.0.0:8000/act" 

# for inference 
class VLAInference:
    def __init__(self, model='openvla/openvla-7b', max_images=1, **kwargs):
        if isinstance(model, str):
            self.model = model
        else:
            raise TypeError("model must be a string")
        
        # extracted from the model.config.json from hugging face
        self.n_action_bins = 256
        self.vocab_size = 32064
        self.pad_to_multiple_of = 64
        self.dof = 7
        
        self.instruction = kwargs.get("instruction", 'stop')
        self.prompt_template = "In: What action should the robot take to ${INSTRUCTION}?\nOut:▁"

        self.max_images = max_images
        
        self.action_spaces = {} 
        self.action_spaces["normalized"] = dict(
            action = dict(
                normalized = True,
                mask = [False] * self.dof,
                qo1 = [-1.0] * self.dof,
                q99 = [1.0] * self.dof,
            )
        )

        for key, space in self.action_spaces.items():
            action = AttributeDict(space)
            action.name = key
            self.action_spaces[key] = action
            for k, v in action.items():
                if isinstance(v, list):
                    action[k] = np.array(v)

        # map the tokenizer vocab range to the discrete action bins ~ 256 
        self.bins = np.linspace(-1, 1, self.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size  = self.vocab_size - self.pad_to_multiple_of
        
    def __call__(self, image, ** kwargs):   
        return self.predict_action(image, **kwargs)
    
    def predict_action(self, image, instruction= '', **kwargs):
        '''
        Given the image and instruction, predict the next action
        '''
        action = None 
        while action is None:
            action = requests.post(SERVER_LOC, json = {"image": image, "instruction": instruction}).json()
        return action
    
    def decode_action(self, token):
        if not token:
            return None
        num_tokens = min(len(token), self.dof)
        tokens = token[:num_tokens]
        
        # map from the vocab bins back into the action space (-1,1)
        # subtract from the vocab size, since the action tokens were placed at the end of the vocab during training
        pred_actions = self.vocab_size - np.array(tokens) 
        pred_actions = np.clip(pred_actions-1, a_min=0, a_max=self.bin_centers[0]-1)
        pred_actions = self.bin_centers[pred_actions]
        
        #TODO: Add denormalization feature in case your robot requires a range different from (-1,1)
        
        return pred_actions
    

    