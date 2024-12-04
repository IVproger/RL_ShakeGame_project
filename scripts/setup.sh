#!/bin/bash 
# Create a virtual environment and install the dependencies
python -m venv .venv
source .venv/bin/activate
pip install -Ur requirements.txt