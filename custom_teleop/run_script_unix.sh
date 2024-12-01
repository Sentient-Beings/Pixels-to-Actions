#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_NAME=$(basename "$SCRIPT_DIR")

reset
make clean
make

./"$PROJECT_NAME"