import re
import json
import torch
import torchvision.transforms as T

from typing import Dict, List
import os


def load_architectures_from_config(config_path: str) -> List[str]:
    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} doesn't exist.")
    # load architectures from config.json
    with open(config_path, 'r') as f:
        config = json.load(f)
        architectures = config.get('architectures', None)
        if architectures is None:
            raise ValueError(f"Architectures not found in {config_path}.")
        if len(architectures) != 1:
            raise ValueError(f"Architectures should have only one element, got {len(architectures)}.")
        model_arch = architectures[0]
    return model_arch

def transform_pixel_values(pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
    # NOTE: this function doesn't accept unbatched inputs
    # pixel_values should be uint8 of (B, T, C, H, W)
    if isinstance(pixel_values, list):
        pixel_values = torch.stack(pixel_values)

    if pixel_values.ndim == 4:
        # pixel_values is (B, C, H, W)
        # (B, C, H, W) -> (B, 1, C, H, W)
        pixel_values = pixel_values.unsqueeze(1)
    elif pixel_values.ndim == 5:
        # pixel_values is (B, T, C, H, W)
        pass
    else:
        raise ValueError(f"pixel_values should be 4D or 5D, got {pixel_values.ndim}D")
    return pixel_values

EOL_PROMPTS = {
    'text': '<sent>\nSummary above sentence in one word:',
    'image': '<image>\nSummary above image in one word:',
    'video': '<video>\nSummary above video in one word:',
}