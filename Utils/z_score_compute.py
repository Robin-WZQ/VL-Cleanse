import torch
from transformers import AutoProcessor, CLIPModel
from transformers.image_utils import load_image
import os
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("/mnt/sdb1/wangzhongqi/Models/clip-vit-large-patch14-336")
processor = AutoProcessor.from_pretrained("/mnt/sdb1/wangzhongqi/Models/clip-vit-large-patch14-336")

backdoor_attack_methods = [
    'badencoder',
    'badnet',
    'BadVision',
    'blended',
    'DRUPE',
    'PBCL',
    'refool',
    'WB'
]

mapping = {
    'badencoder': 'BadEncoder',
    'badnet': 'BadNet',
    'BadVision': 'BadVision',
    'blended': 'Blended',
    'DRUPE': 'DRUPE',
    'PBCL': 'PBCL',
    'refool': 'Refool',
    'WB': 'WB'
}

# ===== 计算dataset feature =====
def extract_features(model, folder, batch_size=64):

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    image_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(valid_ext)
    ]

    print(f"Total images in {folder}: {len(image_paths)}")

    all_features = []

    for i in tqdm(range(0, len(image_paths), batch_size)):

        batch_paths = image_paths[i:i+batch_size]

        images = [load_image(p) for p in batch_paths]

        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = F.normalize(features, dim=-1)

        all_features.append(features)

    all_features = torch.cat(all_features, dim=0)

    print("Feature shape:", all_features.shape)

    return all_features


# ===== 计算 similarity mean / var =====
def compute_similarity_stats(features):

    print("Computing similarity matrix...")

    sim_matrix = features @ features.T   # cosine similarity

    mean = torch.mean(sim_matrix)
    var = torch.var(sim_matrix, unbiased=False)

    return mean.item(), var.item()

clean_folder = "/mnt/sdb1/wangzhongqi/project/backdoor_detection/VL-Cleanse/Data/Test_data/Benign_CLIP"



z_socre_output_file = "/mnt/sdb1/wangzhongqi/project/backdoor_detection/VL-Cleanse/Evaluation/Z_score/z_score_results.txt"

with open(z_socre_output_file, "w") as f:
    for method in tqdm(backdoor_attack_methods):
        print(f"\n===== Processing {method} =====")
        for backdoor_id in os.listdir(f'/mnt/sdb1/wangzhongqi/project/backdoor_detection/VL-Cleanse/Models/Backdoor_models/CLIP-L/{method}'):
            if backdoor_id.startswith('poisoned_model'):
                if int(backdoor_id.split('_')[-1]) > 10:
                    continue
                print(f'### Processing {method} with id {backdoor_id} ###')
                ckpt_path = f'/mnt/sdb1/wangzhongqi/project/backdoor_detection/VL-Cleanse/Models/Backdoor_models/CLIP-L/{method}/{backdoor_id}/pytorch_model.bin'
                state_dict = torch.load(ckpt_path, map_location="cpu")

                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("vision_model."):
                        new_k = k.replace("vision_model.", "")
                        new_state_dict[new_k] = v

                model.vision_model.load_state_dict(new_state_dict)

                model = model.to(device)
                model.eval()

                bad_folder = "/mnt/sdb1/wangzhongqi/project/backdoor_detection/VL-Cleanse/Data/Test_data/" + mapping[method]
            
                clean_features = extract_features(model, clean_folder)

                clean_mean, clean_var = compute_similarity_stats(clean_features)

                bad_features = extract_features(model, bad_folder)

                bad_mean, bad_var = compute_similarity_stats(bad_features)

                # ===== 计算 z-score =====
                zscore = (bad_mean - clean_mean) / clean_var

                f.write(f"{method}_{backdoor_id}: {zscore:.4f}\n")