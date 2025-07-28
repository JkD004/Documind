import json
import os
from datetime import datetime

def load_json(filepath):
    """Loads a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """Saves data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_timestamp():
    """Returns the current timestamp in ISO format."""
    return datetime.now().isoformat()

def clean_text(text):
    """Basic text cleaning for NLP processing."""
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    # You can add more cleaning rules if necessary (e.g., remove punctuation, numbers)
    return text