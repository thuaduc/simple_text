"""Configuration management for the text simplification project."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


# Model Configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'Qwen/Qwen3.5-2B')

# Model Parameters
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', '256'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
LOAD_IN_4BIT = os.getenv('LOAD_IN_4BIT', 'false').lower() == 'true'

# Data Configuration
DATA_DIR = os.getenv('DATA_DIR', 'cochrane/data')

# Evaluation Configuration
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '8'))
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))


def get_config():
    """Get configuration as a dictionary."""
    return {
        'model_name': MODEL_NAME,
        'max_new_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE,
        'load_in_4bit': LOAD_IN_4BIT,
        'data_dir': DATA_DIR,
        'batch_size': BATCH_SIZE,
        'random_seed': RANDOM_SEED,
    }


def print_config():
    """Print current configuration."""
    config = get_config()
    print("Current Configuration:")
    print("=" * 50)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    print_config()
