import cv2
import numpy as np
from vla import VLAInference
import logging

def test_vla_server():
    '''
    This script is used to test the VLA server, run this before executing the pixel_to_action.py script.
    '''
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        image_path = "box.png"
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.asarray(image) 
        
        vla_model = VLAInference()

        if not vla_model.check_server_connection():
            logger.error("Server is not accessible. Please ensure the server is running.")
            return

        instruction = "move to box"
        predicted_actions = vla_model.predict_action(image, instruction=instruction)

        logger.info(f"Instruction: {instruction}")
        logger.info(f"Predicted actions: {predicted_actions}")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")

if __name__ == "__main__":
    test_vla_server()