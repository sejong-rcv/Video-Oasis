#!/usr/bin/env python3
"""
NoSense QA Ours: Multi-Model Zero-shot Video Question Answering

Supports multiple vision-language models:
- CLIP (OpenAI)
- SigLIP (Google)
- SigLIP2 (Google, 2025)
- EVA-CLIP-8B (BAAI)

Experiment configurations:
- Aggregation: mean, max
- Top-K: 32, 128

Usage:
    # Single experiment
    python nosense_qa_ours.py --model siglip2-giant --top_k 32 --aggregation mean

    # Run all experiments
    python nosense_qa_ours.py --run_all --gpus 0,1,2,3

    # Extract features only
    python nosense_qa_ours.py --extract_features_only --model siglip2-giant
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import longclip


# ============================================================
# Model Registry
# ============================================================
@dataclass
class ModelConfig:
    """Configuration for a vision-language model."""
    name: str
    hf_model_id: str
    model_type: str  # "clip", "siglip", "siglip2", "eva"
    feature_dim: int
    requires_timm: bool = False
    requires_eva_clip: bool = False


MODEL_REGISTRY = {
    # OpenAI CLIP models
    "clip-vit-l-14": ModelConfig(
        name="clip-vit-l-14",
        hf_model_id="openai/clip-vit-large-patch14",
        model_type="clip",
        feature_dim=768,
    ),

    # LongCLIP models
    "longclip": ModelConfig(
        name="longclip",
        hf_model_id="../static/checkpoints/LongCLIP-L/longclip-L.pt",
        model_type="longclip",
        feature_dim=768,
    ),

    # Google SigLIP models
    "siglip-l": ModelConfig(
        name="siglip-l",
        hf_model_id="google/siglip-large-patch16-384",
        model_type="siglip",
        feature_dim=1024,
    ),

    # Google SigLIP2 models (2025)
    "siglip2-giant": ModelConfig(
        name="siglip2-giant",
        hf_model_id="google/siglip2-giant-opt-patch16-384",
        model_type="siglip2",
        feature_dim=1536,
    ),

    # BAAI EVA-CLIP models
    "eva-clip-8b": ModelConfig(
        name="eva-clip-8b",
        hf_model_id="BAAI/EVA-CLIP-8B",
        model_type="eva",
        feature_dim=1280,
        requires_eva_clip=True,
    ),

}

# Default experiment configurations
DEFAULT_TOP_K = [32, 128]
DEFAULT_AGGREGATIONS = ["mean", "max"]
DEFAULT_MODELS = ["clip-vit-l-14", "siglip-l", "siglip2-giant", "eva-clip-8b", "longclip"]


# ============================================================
# Model Loading Functions
# ============================================================
def load_clip_model(config: ModelConfig, device: str = "cuda"):
    """Load OpenAI CLIP or Google SigLIP model using transformers."""
    from transformers import AutoProcessor, AutoModel, AutoTokenizer

    print(f"Loading {config.model_type.upper()} model: {config.hf_model_id}")

    processor = AutoProcessor.from_pretrained(config.hf_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_model_id)
    model = AutoModel.from_pretrained(config.hf_model_id)
    model = model.to(device)
    model.eval()

    return model, processor, tokenizer

def load_longclip_model(config: ModelConfig, device: str = "cuda"):
    print(f"Loading {config.model_type.upper()} model: {config.hf_model_id}")

    model, processor = longclip.load(config.hf_model_id, device=device)

    model = model.to(device)
    model.eval()

    return model, None, processor


def load_siglip2_model(config: ModelConfig, device: str = "cuda"):
    """Load Google SigLIP2 model using transformers."""
    from transformers import AutoProcessor, AutoModel, AutoTokenizer

    print(f"Loading {config.model_type.upper()} model: {config.hf_model_id}")

    tokenizer = AutoTokenizer.from_pretrained(config.hf_model_id)
    model = AutoModel.from_pretrained(config.hf_model_id)
    model = model.to(device)
    model.eval()

    return model, None, tokenizer

def load_eva_clip_model(config: ModelConfig, device: str = "cuda"):
    """
    Load EVA-CLIP-8B model.

    EVA-CLIP requires special handling - it uses the eva_clip library
    or can be loaded via open_clip/timm.
    """
    try:
        # Try loading via open_clip first (recommended)
        import open_clip

        print(f"Loading EVA-CLIP model via open_clip: {config.name}")

        if "8b" in config.name.lower():
            # EVA-CLIP-8B
            model, _, preprocess = open_clip.create_model_and_transforms(
                'EVA02-E-14-plus',
                pretrained='laion2b_s9b_b144k',
                device=device,
            )
            tokenizer = open_clip.get_tokenizer('EVA02-E-14-plus')
        else:
            # Fallback to EVA02-L
            model, _, preprocess = open_clip.create_model_and_transforms(
                'EVA02-L-14',
                pretrained='merged2b_s4b_b131k',
                device=device,
            )
            tokenizer = open_clip.get_tokenizer('EVA02-L-14')

        model.eval()
        return model, preprocess, tokenizer, "open_clip"

    except ImportError:
        print("open_clip not found, trying transformers...")

    # Fallback: Load via transformers (may have limited functionality)
    from transformers import AutoProcessor, AutoModel, AutoTokenizer

    print(f"Loading EVA-CLIP model via transformers: {config.hf_model_id}")

    model = AutoModel.from_pretrained(config.hf_model_id, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    model = AutoModel.from_pretrained(config.hf_model_id, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    return model, None, processor, "transformers"


def load_model(model_name: str, device: str = "cuda"):
    """
    Universal model loader.

    Returns:
        model: The loaded model
        processor: Image processor/transform
        tokenizer: Text tokenizer
        config: ModelConfig
        backend: str ("transformers" or "open_clip")
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[model_name]
    if config.model_type == "eva":
        model, processor, tokenizer, backend = load_eva_clip_model(config, device)
        return model, processor, tokenizer, config, backend
    elif config.model_type == "siglip2":
        model, processor, tokenizer = load_siglip2_model(config, device)
        return model, processor, tokenizer, config, "transformers"
    elif config.model_type == "longclip":
        model, processor, tokenizer = load_longclip_model(config, device)
        return model, processor, tokenizer, config, "longclip"
    else:  # clip, siglip
        model, processor, tokenizer = load_clip_model(config, device)
        return model, processor, tokenizer, config, "transformers"


# ============================================================
# Text Encoding Functions
# ============================================================
def encode_text_transformers(
    texts: List[str],
    model,
    tokenizer,
    device: str = "cuda",
    max_length: int = 64,
) -> torch.Tensor:
    """Encode text using transformers-based models (CLIP, SigLIP, SigLIP2)."""

    inputs = tokenizer(
        text=texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        try:
            text_features = model.get_text_features(**inputs)
        except:
            with torch.amp.autocast("cuda"):
                text_features = model.encode_text(**inputs)
        text_features = F.normalize(text_features, dim=-1)

    return text_features


def encode_text_open_clip(
    texts: List[str],
    model,
    tokenizer,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode text using open_clip-based models (EVA-CLIP)."""
    import open_clip

    tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

    return text_features


def encode_text_longclip(
    texts: List[str],
    model,
    tokenizer,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode text using open_clip-based models (Long-CLIP)."""
    tokens = longclip.tokenize(texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

    return text_features

def encode_text(
    texts: List[str],
    model,
    tokenizer,
    device: str = "cuda",
    backend: str = "transformers",
    max_length: int = 64,
) -> torch.Tensor:
    """Universal text encoder."""
    if backend == "open_clip":
        return encode_text_open_clip(texts, model, tokenizer, device)
    elif backend == 'longclip':
        return encode_text_longclip(texts, model, tokenizer, device)
    else:
        return encode_text_transformers(texts, model, tokenizer, device, max_length)


# ============================================================
# Core QA Functions
# ============================================================
def compute_similarity(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity (assumes normalized inputs)."""
    keys = keys.to(query.dtype)
    return torch.matmul(query, keys.T)


def select_topk_frames(
    question_feature: torch.Tensor,
    video_features: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select top-k frames most similar to the question."""
    if question_feature.dim() == 1:
        question_feature = question_feature.unsqueeze(0)

    question_feature = question_feature.to(video_features.dtype)
    similarity = compute_similarity(question_feature, video_features).squeeze(0)
    actual_k = min(top_k, len(video_features))
    topk_values, topk_indices = torch.topk(similarity, actual_k)
    selected_features = video_features[topk_indices]

    return selected_features, topk_indices


def aggregate_features(features: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """
    Aggregate multiple frame features.

    Args:
        features: (K, D) frame features
        method: "mean" or "max"

    Returns:
        For "mean": (D,) aggregated feature
        For "max": (K, D) all features (aggregation done at similarity level)
    """
    if method == "mean":
        aggregated = features.mean(dim=0)
        aggregated = F.normalize(aggregated, dim=-1)
        return aggregated
    elif method == "max":
        return features  # Return all, max pooling done at similarity level
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def predict_answer(
    question: str,
    options: List[str],
    video_features: torch.Tensor,
    model,
    tokenizer,
    device: str = "cuda",
    backend: str = "transformers",
    top_k: int = 32,
    aggregation: str = "mean",
) -> Tuple[int, List[float], Dict]:
    """
    Predict answer for a question.

    Returns:
        predicted_idx: Index of predicted answer
        similarities: Similarity scores for each option
        metadata: Additional information (selected frames, etc.)
    """
    # Encode question
    question_feature = encode_text([question], model, tokenizer, device, backend)

    # Move video features to device
    video_features = video_features.to(device)

    # Select top-k frames based on question similarity
    selected_features, selected_indices = select_topk_frames(
        question_feature, video_features, top_k
    )

    # Aggregate frame features
    aggregated = aggregate_features(selected_features, aggregation)

    # Encode options
    option_features = encode_text(options, model, tokenizer, device, backend)

    # Compute similarities
    if aggregation == "max":
        # Max pooling: for each option, take max similarity across all selected frames
        similarities = compute_similarity(aggregated, option_features)  # (K, num_options)
        max_similarities, _ = similarities.max(dim=0)  # (num_options,)
        predicted_idx = max_similarities.argmax().item()
        similarity_scores = max_similarities.tolist()
    else:
        # Mean aggregation: single aggregated feature vs options
        similarities = compute_similarity(
            aggregated.unsqueeze(0), option_features
        ).squeeze(0)  # (num_options,)
        predicted_idx = similarities.argmax().item()
        similarity_scores = similarities.tolist()

    metadata = {
        "selected_frame_indices": selected_indices.tolist(),
        "num_frames_used": len(selected_indices),
    }

    return predicted_idx, similarity_scores, metadata


# ============================================================
# Utility Functions
# ============================================================
def get_video_id(video_path: str) -> str:
    """Extract video ID from path."""
    return video_path.split('/')[-1]


def index_to_answer(idx: int) -> str:
    """Convert index to answer letter (0 -> A, 1 -> B, etc.)."""
    return chr(ord('A') + idx)


def answer_to_index(answer: str) -> int:
    """Convert answer letter to index (A -> 0, B -> 1, etc.)."""
    return ord(answer.upper()) - ord('A')


def split_data_into_chunks(data: list, num_chunks: int, chunk_id: int) -> list:
    """Split data for multi-GPU processing."""
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    start_idx = chunk_id * chunk_size + min(chunk_id, remainder)
    end_idx = start_idx + chunk_size + (1 if chunk_id < remainder else 0)
    return data[start_idx:end_idx]


# ============================================================
# Main QA Runner
# ============================================================
def run_qa(
    data: List[Dict],
    features_dir: str,
    model,
    tokenizer,
    device: str = "cuda",
    backend: str = "transformers",
    top_k: int = 32,
    aggregation: str = "mean",
    gpu_id: int = 0,
) -> Dict:
    """
    Run QA evaluation on dataset.

    Returns:
        Dictionary with predictions and statistics
    """
    predictions = []
    correct = 0
    total = 0
    missing_features = 0

    desc = f"[GPU {gpu_id}] QA (k={top_k}, agg={aggregation})"

    for item in tqdm(data, desc=desc):
        video_path = item["video_path"]
        video_id = get_video_id(video_path)
        db = item['db']
        question = item["question"]
        options = item["options"]
        ground_truth = item["answer"]
        gt_idx = answer_to_index(ground_truth)
        
        # Load video features 우리 포맷으로 
        feature_path = os.path.join(features_dir, f"{db}**@@**{video_id}.pt")
        
        if not os.path.exists(feature_path):
            missing_features += 1
            continue

        try:
            video_features = torch.load(feature_path, map_location="cpu")

            predicted_idx, similarities, metadata = predict_answer(
                question=question,
                options=options,
                video_features=video_features,
                model=model,
                tokenizer=tokenizer,
                device=device,
                backend=backend,
                top_k=top_k,
                aggregation=aggregation,
            )

            predicted_answer = index_to_answer(predicted_idx)
            is_correct = predicted_answer == ground_truth

            if is_correct:
                correct += 1
            total += 1

            predictions.append({
                "qid": item.get("qid", ""),
                "video_id": video_id,
                "question": question,
                "options": options,
                "ground_truth": ground_truth,
                "predicted": predicted_answer,
                "is_correct": is_correct,
                "similarities": similarities,
                "metadata": metadata,
            })

        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {video_id}: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0

    return {
        "predictions": predictions,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "missing_features": missing_features,
        "config": {
            "top_k": top_k,
            "aggregation": aggregation,
        }
    }


# ============================================================
# Experiment Runner
# ============================================================
def run_single_experiment(
    model_name: str,
    data_path: str,
    features_dir: str,
    output_dir: str,
    top_k: int,
    aggregation: str,
    gpu_id: int = 0,
    num_chunks: int = 1,
    chunk_id: int = 0,
):
    """Run a single experiment configuration."""
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    # Load data
    print(f"[GPU {gpu_id}] Loading data from: {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)

    # Split for multi-GPU
    if num_chunks > 1:
        data = split_data_into_chunks(data, num_chunks, chunk_id)
        print(f"[GPU {gpu_id}] Processing chunk {chunk_id}/{num_chunks} ({len(data)} samples)")

    # Load model
    print(f"[GPU {gpu_id}] Loading model: {model_name}")
    model, processor, tokenizer, config, backend = load_model(model_name, device)
    print(f"[GPU {gpu_id}] Model loaded: {config.hf_model_id} (backend: {backend})")

    # Run QA
    print(f"[GPU {gpu_id}] Running QA: top_k={top_k}, aggregation={aggregation}")
    results = run_qa(
        data=data,
        features_dir=features_dir,
        model=model,
        tokenizer=tokenizer,
        device=device,
        backend=backend,
        top_k=top_k,
        aggregation=aggregation,
        gpu_id=gpu_id,
    )

    # Add model info to results
    results["model"] = {
        "name": model_name,
        "hf_model_id": config.hf_model_id,
        "model_type": config.model_type,
        "backend": backend,
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"[GPU {gpu_id}] Results Summary")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Top-K: {top_k}, Aggregation: {aggregation}")
    print(f"Total: {results['total']}, Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    if results['missing_features'] > 0:
        print(f"Missing features: {results['missing_features']}")
    print("=" * 60)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"qa_results_{model_name}_k{top_k}_{aggregation}"
    if num_chunks > 1:
        output_filename += f"_chunk{chunk_id}"
    output_path = os.path.join(output_dir, f"{output_filename}.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[GPU {gpu_id}] Results saved to: {output_path}")

    return results


def run_all_experiments(
    data_path: str,
    base_features_dir: str,
    output_dir: str,
    gpus: List[int],
    models: List[str] = None,
    top_ks: List[int] = None,
    aggregations: List[str] = None,
):
    """
    Run all experiment configurations.

    This creates a grid of experiments:
    - Models: clip-vit-l-14, siglip-so400m, siglip2-giant, eva-clip-8b
    - Top-K: 32, 128
    - Aggregation: mean, max
    """
    models = models or DEFAULT_MODELS
    top_ks = top_ks or DEFAULT_TOP_K
    aggregations = aggregations or DEFAULT_AGGREGATIONS

    # Generate all experiment configurations
    experiments = []
    for model_name in models:
        config = MODEL_REGISTRY.get(model_name)
        if config is None:
            print(f"Warning: Unknown model {model_name}, skipping...")
            continue

        # Determine features directory based on model type
        features_dir = os.path.join(base_features_dir, f"{config.model_type}")

        for top_k in top_ks:
            for agg in aggregations:
                experiments.append({
                    "model_name": model_name,
                    "features_dir": features_dir,
                    "top_k": top_k,
                    "aggregation": agg,
                })

    print(f"Total experiments to run: {len(experiments)}")
    print(f"Models: {models}")
    print(f"Top-K values: {top_ks}")
    print(f"Aggregation methods: {aggregations}")
    print(f"GPUs: {gpus}")
    print()

    # Run experiments (simple sequential for now, can be parallelized)
    all_results = []
    for i, exp in enumerate(experiments):
        gpu_id = gpus[i % len(gpus)]
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(experiments)}")
        print(f"Model: {exp['model_name']}, K: {exp['top_k']}, Agg: {exp['aggregation']}")
        print(f"{'='*60}")

        try:
            results = run_single_experiment(
                model_name=exp["model_name"],
                data_path=data_path,
                features_dir=exp["features_dir"],
                output_dir=output_dir,
                top_k=exp["top_k"],
                aggregation=exp["aggregation"],
                gpu_id=gpu_id,
            )
            all_results.append({
                "experiment": exp,
                "accuracy": results["accuracy"],
                "total": results["total"],
            })
        except Exception as e:
            print(f"Error in experiment: {e}")
            all_results.append({
                "experiment": exp,
                "error": str(e),
            })

    # Print summary table
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Top-K':<8} {'Agg':<8} {'Accuracy':<12} {'Total':<8}")
    print("-" * 80)
    for r in all_results:
        exp = r["experiment"]
        if "error" in r:
            print(f"{exp['model_name']:<20} {exp['top_k']:<8} {exp['aggregation']:<8} {'ERROR':<12}")
        else:
            print(f"{exp['model_name']:<20} {exp['top_k']:<8} {exp['aggregation']:<8} {r['accuracy']:.2%}       {r['total']:<8}")
    print("=" * 80)

    # Save summary
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


# ============================================================
# Feature Extraction
# ============================================================
def extract_features_for_model(
    model_name: str,
    data_path: str,
    frames_dir: str,
    output_dir: str,
    gpu_id: int = 0,
    batch_size: int = 32,
):
    """
    Extract video features using specified model.

    This loads frames and encodes them using the vision encoder.
    """
    from PIL import Image

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    # Load model
    model, processor, tokenizer, config, backend = load_model(model_name, device)
    print(f"Loaded model: {config.hf_model_id}")

    # Load data
    with open(data_path, "r") as f:
        data = json.load(f)

    # Get unique videos
    video_ids = list(set(get_video_id(item["video_path"]) for item in data))
    print(f"Total unique videos: {len(video_ids)}")

    os.makedirs(output_dir, exist_ok=True)

    for video_id in tqdm(video_ids, desc="Extracting features"):
        output_path = os.path.join(output_dir, f"{video_id}.pt")
        if os.path.exists(output_path):
            continue

        # Find frames for this video
        video_frames_dir = os.path.join(frames_dir, video_id)
        if not os.path.exists(video_frames_dir):
            print(f"Warning: No frames found for {video_id}")
            continue

        # Load frames
        frame_files = sorted([
            f for f in os.listdir(video_frames_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])

        if not frame_files:
            print(f"Warning: No frame files in {video_frames_dir}")
            continue

        # Process frames in batches
        all_features = []

        for i in range(0, len(frame_files), batch_size):
            batch_files = frame_files[i:i+batch_size]
            images = []

            for f in batch_files:
                img_path = os.path.join(video_frames_dir, f)
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue

            if not images:
                continue

            # Encode images
            with torch.no_grad():
                if backend == "open_clip":
                    # open_clip uses preprocess function
                    inputs = torch.stack([processor(img) for img in images]).to(device)
                    features = model.encode_image(inputs)
                else:
                    # transformers uses processor
                    inputs = processor(images=images, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    features = model.get_image_features(**inputs)

                features = F.normalize(features, dim=-1)
                all_features.append(features.cpu())

        if all_features:
            video_features = torch.cat(all_features, dim=0)
            torch.save(video_features, output_path)

    print(f"Feature extraction complete. Saved to: {output_dir}")


# ============================================================
# Main Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="NoSense QA Ours: Multi-Model Zero-shot Video QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  python nosense_qa_ours.py --model siglip2-giant --top_k 32 --aggregation mean

  # Run all experiments
  python nosense_qa_ours.py --run_all --gpus 0,1,2,3

  # Run specific models only
  python nosense_qa_ours.py --run_all --models clip-vit-l-14,siglip2-giant

  # Extract features
  python nosense_qa_ours.py --extract_features_only --model siglip2-giant

Available models:
  CLIP:    clip-vit-b-32, clip-vit-b-16, clip-vit-l-14, clip-vit-l-14-336
  SigLIP:  siglip-base, siglip-large, siglip-so400m
  SigLIP2: siglip2-base, siglip2-large, siglip2-so400m, siglip2-giant
  EVA:     eva-clip-8b, eva-clip-8b-448
        """
    )

    # Mode selection
    parser.add_argument("--run_all", action="store_true",
                        help="Run all experiment configurations")
    parser.add_argument("--extract_features_only", action="store_true",
                        help="Only extract features, don't run QA")

    # Data paths
    parser.add_argument("--data_path", type=str,
                        default="/data3/sjpark/workspace/LVU_release/data/videomme_all.json",
                        help="Path to benchmark JSON file")
    parser.add_argument("--features_dir", type=str,
                        default="NoSense/data",
                        help="Base directory for features (model-specific subdirs)")
    parser.add_argument("--frames_dir", type=str,
                        default="NoSense/data/frames",
                        help="Directory containing extracted frames")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/exp_ours",
                        help="Output directory for results")

    # Model configuration
    parser.add_argument("--model", type=str, default="siglip2-giant",
                        help="Model to use (see --help for list)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of models for --run_all")

    # Experiment configuration
    parser.add_argument("--top_k", type=int, default=32,
                        help="Number of frames to select")
    parser.add_argument("--top_ks", type=str, default="32,128",
                        help="Comma-separated top-k values for --run_all")
    parser.add_argument("--aggregation", type=str, default="mean",
                        choices=["mean", "max"],
                        help="Feature aggregation method")
    parser.add_argument("--aggregations", type=str, default="mean,max",
                        help="Comma-separated aggregation methods for --run_all")

    # GPU configuration
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs for --run_all")
    parser.add_argument("--num_chunks", type=int, default=1,
                        help="Number of data chunks (for multi-GPU)")
    parser.add_argument("--chunk_id", type=int, default=0,
                        help="Chunk ID to process")

    # Other options
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for feature extraction")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: process only 10 samples")

    args = parser.parse_args()

    # Parse comma-separated arguments
    gpus = [int(g) for g in args.gpus.split(",")]
    top_ks = [int(k) for k in args.top_ks.split(",")]
    aggregations = args.aggregations.split(",")
    models = args.models.split(",") if args.models else None

    # Feature extraction mode
    if args.extract_features_only:
        config = MODEL_REGISTRY.get(args.model)
        if config is None:
            print(f"Unknown model: {args.model}")
            return

        output_dir = os.path.join(args.features_dir, f"{config.model_type}")
        extract_features_for_model(
            model_name=args.model,
            data_path=args.data_path,
            frames_dir=args.frames_dir,
            output_dir=output_dir,
            gpu_id=args.gpu_id,
            batch_size=args.batch_size,
        )
        return

    # Run all experiments mode
    if args.run_all:
        run_all_experiments(
            data_path=args.data_path,
            base_features_dir=args.features_dir,
            output_dir=args.output_dir,
            gpus=gpus,
            models=models,
            top_ks=top_ks,
            aggregations=aggregations,
        )
        return

    # Single experiment mode
    config = MODEL_REGISTRY.get(args.model)
    if config is None:
        print(f"Unknown model: {args.model}")
        print(f"Available models: {list(MODEL_REGISTRY.keys())}")
        return
    
    features_dir = os.path.join(args.features_dir, f"{config.model_type}")

    run_single_experiment(
        model_name=args.model,
        data_path=args.data_path,
        features_dir=features_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        aggregation=args.aggregation,
        gpu_id=args.gpu_id,
        num_chunks=args.num_chunks,
        chunk_id=args.chunk_id,
    )


if __name__ == "__main__":
    main()
