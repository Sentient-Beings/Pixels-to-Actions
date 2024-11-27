#! /bin/bash

# Get the project root directory (one level up from src)
export ROBOT_LEARNING_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Add the project root to PYTHONPATH
export PYTHONPATH=$ROBOT_LEARNING_ROOT:$PYTHONPATH

# Set the model path for the Panda robot
export MJ_PANDA_PATH=$ROBOT_LEARNING_ROOT/model

# Print the environment setup
echo -e "Environment setup complete:"
echo -e "ROBOT_LEARNING_ROOT=$ROBOT_LEARNING_ROOT"
echo -e "MJ_PANDA_PATH=$MJ_PANDA_PATH\n"
echo -e "Python path updated to include robot_learning package."
