import argparse
import os
import random
import json
import tarfile
import io
from pathlib import Path

from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent


def relative_path(*parts):
    return os.path.relpath(SCRIPT_DIR.joinpath(*parts), start=Path.cwd())


def create_parser():
    parser = argparse.ArgumentParser(description='Benign fine-tuning of the vision encoder on an image-text dataset')
    parser.add_argument('--cc3m_root', type=str, default=relative_path('..', '..', '..', '..', 'Dataset', 'CC3M'), help='root directory of the dataset archives')
    parser.add_argument('--attack', type=str, choices=['BadVision', 'BadVision_untargeted', 'badencoder','PBCL','WB','DRUPE','badnet','sig','blended','wanet','refool'], default='badnet', required=False, help='attack type')
    parser.add_argument('--split', type=str, choices=['train', 'validation'], default='train', help='dataset split')
    parser.add_argument('--train_tars', type=int, default=576, help='number of training archives')
    parser.add_argument('--val_tars', type=int, default=16, help='number of validation archives')
    parser.add_argument('--max_samples', type=int, default=100, help='maximum number of samples to load to avoid out-of-memory errors')
    parser.add_argument('--cc3m_list_json', type=str, default='', help='optional prefiltered sample list')
    parser.add_argument('--model_name', type=str, default=relative_path('..', '..', '..', '..', 'Models', 'clip-vit-large-patch14-336'), help='path to the base vision-language model')
    parser.add_argument('--backdoor_model_root', type=str, default=relative_path('..', 'Models', 'Backdoor_models', 'CLIP-L-336'), help='root directory of backdoored models')
    parser.add_argument('--batch_size', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=4.5e-6, help='fine-tuning learning rate')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0', help='device used for execution')
    parser.add_argument('--save_path', type=str, default=relative_path('..', 'Mitigation', 'Post_training', 'VL-Cleanse', 'Models'))
    parser.add_argument('--reversed_path', type=str, default=relative_path('..', 'Results'))
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


class CC3MTarDataset(Dataset):
    def __init__(self, tar_paths=None, cc3m_root=None, split='train', max_samples=None, list_json=None, mask_path=None, patch_path=None):
        self.entries = []  # Each entry stores the archive path, image key, and text key
        self.mask_path = mask_path
        self.patch_path = patch_path
        self.size = (336, 336)

        if list_json:
            with open(list_json, 'r') as f:
                sample_list = json.load(f)
            for item in sample_list:
                tar_path = item.get('tar_path')
                image_key = item.get('key')
                text_key = item.get('key', '').replace('.jpg', '.txt')
                if tar_path and image_key and os.path.exists(tar_path):
                    self.entries.append((tar_path, image_key, text_key))
                    if max_samples and len(self.entries) >= max_samples:
                        break

        else:
            if tar_paths is None:
                if split == 'train':
                    tar_paths = [os.path.join(cc3m_root, f"cc3m-train-{i:04d}.tar") for i in range(576)]
                else:
                    tar_paths = [os.path.join(cc3m_root, f"cc3m-validation-{i:04d}.tar") for i in range(16)]

            for tar_path in tqdm(tar_paths, desc='Enumerating CC3M tars'):
                if not os.path.exists(tar_path):
                    continue
                with tarfile.open(tar_path, 'r') as tar:
                    members = tar.getmembers()
                    image_members = {m.name: m for m in members if m.name.endswith('.jpg')}
                    text_members = {m.name: m for m in members if m.name.endswith('.txt')}
                    for text_name, text_member in text_members.items():
                        image_name = text_name.replace('.txt', '.jpg')
                        if image_name in image_members:
                            self.entries.append((tar_path, image_name, text_name))
                            if max_samples and len(self.entries) >= max_samples:
                                break
                if max_samples and len(self.entries) >= max_samples:
                    break

        if len(self.entries) == 0:
            raise RuntimeError('No samples found in CC3M dataset.')
    
    def reverse_image(self, benign_img):
        transform = transforms.ToTensor()  # Scale pixel values to the range from zero to one

        mask = torch.load(self.mask_path).squeeze(0)

        patch = torch.load(self.patch_path).squeeze(0)
        benign = transform(benign_img)

        reversed_img = benign + mask * patch
        reversed_img = torch.clip(reversed_img, min=0, max=1)

        img = reversed_img.permute(1, 2, 0)*255

        # Convert to an unsigned 8-bit integer array
        img = img.numpy().astype(np.uint8)

        reversed_image = Image.fromarray(img)
        
        return reversed_image

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        tar_path, image_key, text_key = self.entries[idx]
        with tarfile.open(tar_path, 'r') as tar:
            img_member = tar.getmember(image_key)
            img_file = tar.extractfile(img_member)
            image = Image.open(io.BytesIO(img_file.read())).convert('RGB')
            image = image.resize(self.size)
            image = self.reverse_image(image)

            txt_member = tar.getmember(text_key)
            txt_file = tar.extractfile(txt_member)
            caption = txt_file.read().decode('utf-8', errors='ignore').strip()

        return image, caption


def cc3m_collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    return images, texts


def main():
    args = create_parser()
    set_seed(args.seed)

    requested_device = torch.device(args.device)
    if requested_device.type == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = requested_device
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name)
    
    # Load the backdoored model
    suffix = 0
    ckpt_path = os.path.join(
        args.backdoor_model_root,
        args.attack,
        f"poisoned_model_{suffix}",
        "pytorch_model.bin",
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    
    print(f"Loading backdoored model {args.attack} from {ckpt_path}...")
    
    # Remove the vision-model prefix from state-dictionary keys
    if any(k.startswith("vision_model.") for k in state_dict.keys()):
        state_dict = {k.replace("vision_model.", "", 1): v for k, v in state_dict.items()}

    if args.attack.lower() == "pbcl":
        vision_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("visual_projection.")
        }
        projection_state_dict = {
            k.replace("visual_projection.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("visual_projection.")
        }
        model.vision_model.load_state_dict(vision_state_dict)
        model.visual_projection.load_state_dict(projection_state_dict)
    else:
        model.vision_model.load_state_dict(state_dict)

    for param in model.text_model.parameters():
        param.requires_grad = False
    model.logit_scale.requires_grad = False

    model.vision_model.train()
    model.text_model.eval()
    
    mask_path = os.path.join(args.reversed_path, args.attack, f'CLIP-L-336/poisoned_model_{suffix}/Soft_Mask.pt')
    patch_path = os.path.join(args.reversed_path, args.attack, f'CLIP-L-336/poisoned_model_{suffix}/Backdoor_Embedding.pt')

    print("Loading CC3M dataset...")
    tar_count = args.train_tars if args.split == 'train' else args.val_tars
    tar_prefix = 'cc3m-train' if args.split == 'train' else 'cc3m-validation'
    tar_paths = [
        os.path.join(args.cc3m_root, f"{tar_prefix}-{i:04d}.tar")
        for i in range(tar_count)
    ]
    dataset = CC3MTarDataset(
        tar_paths=tar_paths,
        cc3m_root=args.cc3m_root,
        split=args.split,
        max_samples=args.max_samples,
        list_json=args.cc3m_list_json if args.cc3m_list_json else None,
        mask_path=mask_path,
        patch_path=patch_path
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=cc3m_collate_fn, num_workers=4)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.vision_model.parameters()), lr=args.lr, weight_decay=1e-4)

    save_path = os.path.join(args.save_path, args.attack, f"poisoned_model_{suffix}")
    os.makedirs(save_path, exist_ok=True)
    print("Starting CC3M fine-tuning...")

    for epoch in range(args.epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, texts in progress_bar:
            optimizer.zero_grad()

            inputs = processor(images=images, text=texts, return_tensors="pt", padding="max_length", max_length=tokenizer.model_max_length, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            img_features = outputs.image_embeds
            text_features = outputs.text_embeds

            img_embeds = img_features / img_features.norm(dim=-1, keepdim=True)
            txt_embeds = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * img_embeds @ txt_embeds.t()
            logits_per_text = logits_per_image.t()

            batch_size = img_embeds.shape[0]
            labels = torch.arange(batch_size, device=device)

            loss_i = nn.functional.cross_entropy(logits_per_image, labels)
            loss_t = nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        epoch_path = os.path.join(save_path, f'epoch_{epoch+1}')
        os.makedirs(epoch_path, exist_ok=True)
        new_state_dict = {k.replace("vision_tower.", ""): v.detach().cpu() for k, v in model.vision_model.state_dict().items()}
        torch.save(new_state_dict, os.path.join(epoch_path, 'pytorch_model.bin'))

    print("Done!")


if __name__ == '__main__':
    main()
