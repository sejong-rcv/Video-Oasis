import json
import os

def load_video_data(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)