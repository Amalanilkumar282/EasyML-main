# First, create a new file called config.py in your project root
# config.py
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(PROJECT_ROOT, 'saved_models')

# Create the directory if it doesn't exist
os.makedirs(MODEL_FOLDER, exist_ok=True)