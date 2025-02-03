''' 
This script provides a class for VLA inference, the model itself is hosted on a remote server. 
We will be using Runpod to host the model. If you want to run it locally, strip the server calls from this script and link the model locally.
The server deploys the model (through HF AutoClass API) over REST API.
'''
from typing import List, Optional, Union
import logging
import requests
import json_numpy
import numpy as np
from requests.exceptions import RequestException
from cv2 import resize, INTER_LINEAR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

json_numpy.patch()
SERVER_LOC = "http://0.0.0.0:8000/act"

class VLAInference:
    def __init__(self, 
                 model: str = 'openvla/openvla-7b',
                 max_images: int = 1,
                 instruction: str = 'stop',
                 **kwargs):
        """
        Initialize the VLA inference client.
        Args:
            model: Model identifier string
            max_images: Maximum number of images to process
            instruction: Default instruction if none provided
        """
        if not isinstance(model, str):
            raise TypeError("model must be a string")
        
        self.model = model
        self.max_images = max_images
        self.default_instruction = instruction
        
        self.n_action_bins = 256
        self.base_vocab_size = 32064  
        self.padding_size = 64
        self.vocab_size = self.base_vocab_size - self.padding_size
        self.dof = 7
        
        self.prompt_template = "In: What action should the robot take to ${INSTRUCTION}?\nOut:▁"

        self.bins = np.linspace(-1, 1, self.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        
        logger.info(f"Initialized VLAInference with model: {model}")
        
    def format_prompt(self, instruction: str) -> str:
        """Format the prompt with the given instruction."""
        return self.prompt_template.replace("${INSTRUCTION}", instruction)
    
    def check_server_connection(self) -> bool:
        """Check if the server is accessible."""
        try:
            response = requests.get(SERVER_LOC)
            return response.status_code == 200
        except RequestException as e:
            logger.error(f"Server connection failed: {e}")
            return False
        
    def __call__(self, image: np.ndarray, **kwargs) -> List[float]:   
        return self.predict_action(image, **kwargs)
    
    def predict_action(self, 
                      image: np.ndarray, 
                      instruction: str = '', 
                      **kwargs) -> List[float]:
        '''
        Given the image and instruction, predict the next action.
        Args:
            image: Input image as numpy array
            instruction: Text instruction for the action     
        Returns:
            List of predicted action values for each DOF
        '''
        if instruction == '':
            instruction = self.default_instruction
            
        formatted_prompt = self.format_prompt(instruction)
        
        try:
            if not isinstance(image, np.ndarray):
                raise ValueError("Image must be a numpy array")
                
            payload = {
                "image": image,
                "instruction": formatted_prompt
            }
            
            response = requests.post(SERVER_LOC, json=payload)
            response.raise_for_status()
            token_response = response.json()
            
            logger.debug(f"Received response from server: {token_response}")
            return self.decode_action(token_response)
            
        except (RequestException, json.JSONDecodeError) as e:
            logger.error(f"Server error: {e}")
            return [0.0] * self.dof
        except ValueError as e:
            logger.error(f"Input error: {e}")
            return [0.0] * self.dof
    
    def decode_action(self, token: Optional[List[int]]) -> Optional[List[float]]:
        """
        Decode the token response into action values.
        Args:
            token: List of token indices from the model
        Returns:
            List of decoded action values or None if token is invalid
        """
        if not token:
            logger.warning("Received empty token sequence")
            return None
        
        num_tokens = min(len(token), self.dof)
        tokens = token[:num_tokens]
        
        # Map from the vocab bins back into the action space (-1,1)
        bin_indices = self.vocab_size - np.array(tokens) - 1
        bin_indices = np.clip(bin_indices, 0, self.bin_centers.shape[0] - 1)
        pred_actions = self.bin_centers[bin_indices]
        
        logger.debug(f"Decoded actions: {pred_actions}")
        return pred_actions.tolist()
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image to match model requirements (256x256x3).
        
        Args:
            image: Input image as numpy array of any size
        Returns:
            np.ndarray: Resized image of shape (256, 256, 3)
        Raises:
            ValueError: If input image is invalid or doesn't have 3 channels
        """
        try:
            if not isinstance(image, np.ndarray):
                raise ValueError("Input must be a numpy array")
            
            if len(image.shape) != 3:
                raise ValueError("Image must have 3 dimensions (H, W, C)")

            if image.shape[2] != 3:
                raise ValueError("Image must have 3 channels (RGB)")
            
            # Model expects 256x256x3 image
            target_size = (256, 256)
            resized_image = resize(image, target_size, interpolation=INTER_LINEAR)
            resized_image = resized_image.astype(np.float32)
            
            logger.debug(f"Preprocessed image shape: {resized_image.shape}")
            return resized_image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise

    