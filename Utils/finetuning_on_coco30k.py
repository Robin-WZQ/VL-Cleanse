import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

import warnings
warnings.filterwarnings("ignore")

def create_parser():
    parser = argparse.ArgumentParser(description='Benign Fine-tuning of CLIP Text Encoder on COCO')
    parser.add_argument('--coco_img_dir', type=str, default="/mnt/sdb1/wangzhongqi/Dataset/coco2017/images/val2017", help='Path to COCO images directory')
    parser.add_argument('--coco_ann_file', type=str, default="/mnt/sdb1/wangzhongqi/Dataset/coco2017/annotations/captions_val2017.json", help='Path to COCO annotations JSON')
    parser.add_argument('--model_name', type=str, default="/mnt/sdb1/wangzhongqi/Models/LongCLIP-L", help='Base CLIP model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for fine-tuning')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='/mnt/sdb1/wangzhongqi/project/backdoor_detection/VL-Cleanse/Models/Benign_models/LongCLIP')
    return parser.parse_args()

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def custom_collate_fn(batch):
    """
    COCO dataset returns (image, [list_of_captions]).
    For contrastive learning, we randomly sample one caption per image.
    """
    images = [item[0] for item in batch]
    texts = [random.choice(item[1]) for item in batch]
    return images, texts

def main():
    args = create_parser()
    set_seed(args.seed)

    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model, Processor, and Tokenizer
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name)

    # 2. Freeze Text Encoder & Logit Scale, Train Text Encoder
    for param in model.text_model.parameters():
        param.requires_grad = False
    
    # Optional: Freeze the text projection layer if you only want to train the transformer backbone
    # for param in model.text_projection.parameters():
    #     param.requires_grad = False

    model.logit_scale.requires_grad = False # Usually kept frozen during fine-tuning

    model.vision_model.train()
    model.text_model.eval()  # Freeze text encoder

    # 3. Load COCO Dataset
    print("Loading COCO dataset...")
    dataset = CocoCaptions(root=args.coco_img_dir, annFile=args.coco_ann_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=4)

    # 4. Setup Optimizer
    # Only pass parameters that require gradients (Vision Encoder parameters)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.vision_model.parameters()), lr=args.lr, weight_decay=1e-4)

    # 5. Training Loop
    os.makedirs(args.save_path, exist_ok=True)
    
    print("Starting benign fine-tuning...")
    for epoch in range(args.epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, texts in progress_bar:
            optimizer.zero_grad()

            # Process inputs
            inputs = processor(images=images, text=texts, return_tensors="pt", padding="max_length", max_length=tokenizer.model_max_length, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)
            
            # Extract projected features
            img_features = outputs.image_embeds
            text_features = outputs.text_embeds

            # ---------------------------------------------------
            # Adapted Symmetric Contrastive Loss (Cleaned version)
            # ---------------------------------------------------
            # Normalize embeddings
            img_embeds = img_features / img_features.norm(dim=-1, keepdim=True)
            txt_embeds = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity with temperature
            # Using model's built-in logit_scale is usually safer than hardcoding 0.07
            logit_scale = model.logit_scale.exp() 
            logits_per_image = logit_scale * img_embeds @ txt_embeds.t()
            logits_per_text = logits_per_image.t()

            # CLIP-style labels (diagonal)
            batch_size = img_embeds.shape[0]
            labels = torch.arange(batch_size, device=device)

            # Symmetric contrastive loss
            loss_i = nn.functional.cross_entropy(logits_per_image, labels)
            loss_t = nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2
            # ---------------------------------------------------

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        os.makedirs(args.save_path + f'/epoch_{epoch+1}', exist_ok=True)

        new_state_dict = {k.replace("vision_tower.", ""): v.detach().cpu() for k, v in model.vision_model.state_dict().items()}
        
        torch.save(new_state_dict, args.save_path + f'/epoch_{epoch+1}/pytorch_model.bin')
        
    print("Done!")

if __name__ == '__main__':
    main()