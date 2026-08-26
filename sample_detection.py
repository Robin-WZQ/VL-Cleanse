import argparse
import os
from pathlib import Path

import matplotlib

# Use a non-interactive plotting backend for server-side execution
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
from transformers import AutoProcessor, CLIPVisionModel


SCRIPT_DIR = Path(__file__).resolve().parent
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def relative_path(*parts):
    return os.path.relpath(SCRIPT_DIR.joinpath(*parts), start=Path.cwd())


def parse_args():
    parser = argparse.ArgumentParser(description="Detect backdoored samples with a reversed feature")
    parser.add_argument(
        "--base_model_name",
        default=relative_path("..", "..", "..", "..", "Models", "clip-vit-large-patch14-336"),
        help="path to the base vision encoder",
    )
    parser.add_argument(
        "--model_name",
        default=relative_path(
            "..",
            "Models",
            "Backdoor_models",
            "CLIP-L-336",
            "badnet",
            "poisoned_model_0",
            "pytorch_model.bin",
        ),
        help="path to the model checkpoint under evaluation",
    )
    parser.add_argument(
        "--feature_path",
        default=relative_path(
            "..",
            "Results",
            "badnet",
            "CLIP-L-336",
            "poisoned_model_0",
            "Backdoor_Feature.pt",
        ),
        help="path to the reversed backdoor feature",
    )
    parser.add_argument(
        "--benign_folder",
        default=relative_path("..", "Data", "Test_data", "Benign_CLIP"),
        help="directory containing benign test images",
    )
    parser.add_argument(
        "--backdoor_folder",
        default=relative_path("..", "Data", "Test_data", "BadNet"),
        help="directory containing backdoored test images",
    )
    parser.add_argument(
        "--output_path",
        default=relative_path("Results", "sample_detection_roc.png"),
        help="path used to save the detection curve",
    )
    parser.add_argument("--device", default="cuda:7", help="device used for execution")
    parser.add_argument("--batch_size", type=int, default=64, help="inference batch size")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum number of images per class; all images are used by default",
    )
    return parser.parse_args()


def select_device(device_name):
    requested_device = torch.device(device_name)
    if requested_device.type != "cuda":
        return requested_device
    if not torch.cuda.is_available():
        print("No available GPU was detected; falling back to the CPU.")
        return torch.device("cpu")
    if requested_device.index is not None and requested_device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested GPU index {requested_device.index}, "
            f"but only {torch.cuda.device_count()} GPUs are available."
        )
    return requested_device


def collect_image_paths(folder, limit=None):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")

    image_paths = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )
    if limit is not None:
        if limit <= 0:
            raise ValueError("The image limit must be greater than zero.")
        image_paths = image_paths[:limit]
    if not image_paths:
        raise RuntimeError(f"No valid images were found in: {folder}")
    return image_paths


def compute_scores(model, processor, backdoor_feature, image_paths, device, batch_size):
    scores = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="Extracting image features"):
        batch_paths = image_paths[start:start + batch_size]
        images = []
        for image_path in batch_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            image_features = model(**inputs).last_hidden_state.mean(dim=1)
            target_features = backdoor_feature.expand(image_features.shape[0], -1)
            batch_scores = F.cosine_similarity(target_features, image_features, dim=1)
        scores.extend(batch_scores.cpu().tolist())
    return scores


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("The inference batch size must be greater than zero.")

    device = select_device(args.device)
    model = CLIPVisionModel.from_pretrained(args.base_model_name)
    processor = AutoProcessor.from_pretrained(args.base_model_name)
    state_dict = torch.load(args.model_name, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    backdoor_feature = torch.load(args.feature_path, map_location="cpu").to(device)
    if backdoor_feature.ndim == 1:
        backdoor_feature = backdoor_feature.unsqueeze(0)
    if backdoor_feature.ndim != 2 or backdoor_feature.shape[0] != 1:
        raise ValueError("The reversed backdoor feature must be a single two-dimensional feature vector.")

    benign_paths = collect_image_paths(args.benign_folder, args.limit)
    backdoor_paths = collect_image_paths(args.backdoor_folder, args.limit)
    benign_scores = compute_scores(
        model,
        processor,
        backdoor_feature,
        benign_paths,
        device,
        args.batch_size,
    )
    backdoor_scores = compute_scores(
        model,
        processor,
        backdoor_feature,
        backdoor_paths,
        device,
        args.batch_size,
    )

    labels = np.concatenate(
        [np.zeros(len(benign_scores)), np.ones(len(backdoor_scores))]
    )
    scores = np.asarray(benign_scores + backdoor_scores)
    false_positive_rate, true_positive_rate, _ = roc_curve(
        labels,
        scores,
        pos_label=1,
    )
    roc_auc = auc(false_positive_rate, true_positive_rate)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        color="darkorange",
        linewidth=2,
        label=f"ROC curve (AUC = {roc_auc:.4f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", linewidth=2, linestyle="--")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Number of benign samples: {len(benign_scores)}")
    print(f"Number of backdoored samples: {len(backdoor_scores)}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Detection curve saved to: {output_path}")


if __name__ == "__main__":
    main()
