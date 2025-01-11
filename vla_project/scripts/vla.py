''' 
This script provides a class for VLA inference, the model itself is hosted on a remote server. 
We will be using Runpod to host the model. If you want to run it locally, strip the server calls from this script and link the model locally.
The server deploys the model (through HF AutoClass API) over REST API.
'''
# client side specific
import requests
import json_numpy
json_numpy.patch()
import numpy as np 
SERVER_LOC = "http://0.0.0.0:8000/act" 

# for inference 


class VLAInference:
    def __init__(self, model='openvla/openvla-7b', action_space={}, max_images=1, **kwargs):
        if isinstance(model, str):
            self.model = model
        else:
            raise TypeError("model must be a string")
        
        self.instruction = kwargs.get("instruction", 'stop')
        self.prompt_template = "In: What action should the robot take to ${INSTRUCTION}?\nOut:▁"

        self.max_images = max_images

        if action_space is None:
            self.action_space = {}
        if not isinstance(action_space, dict):
            raise TypeError("action_space must be a dictionary but got {}".format(type(action_space)))
        
        action_spaces = {} 
        if (len(action_space) > 0):
            action = action_space[list(action_space.keys())[0]]
            if isinstance(action, dict) and 'action' in action:
                action_spaces = action['action']
                action_space = "normalized"
                
        action_spaces["normalized"] = dict(
            action = dict(
                normalized = True,
                mask = [False] * 7,
                qo1 = [-1.0] * 7,
                q99 = [1.0] * 7,
            )
        )

        for key, space in action_spaces.items():
            action = AttributeDict(space)
            action.name = key
            action_spaces[key] = action
            for k, v in action.items():
                if isinstance(v, list):
                    action[k] = np.array(v)
        
        self.action_spaces = action_spaces
        self.action_space = action_space 

        # map the tokenizer vocab range to the discrete action bins ~ 256 
        self.
    @property
    def dof(self):
        pass
    
    @property
    def action_space(self):
        pass
    
    @action_space.setter
    def action_space(self, key):
        pass 

    def predict_action(self, image):
        pass
    
    def decode_action(self, action):
        pass
    
    def __call__(self, image):
        return self.predict_action(image)  
    