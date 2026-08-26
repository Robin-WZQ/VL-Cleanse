import torch
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel, AutoProcessor
import time
import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from Utils.set_seed import set_random_seed
from Utils.dataset import Feature_Dataset
from Utils.loss import SimilarityLoss, AssimilationLoss, Loss_sim_inter
from test import attack_success_rate_soft, same_target_judge_soft


SCRIPT_DIR = Path(__file__).resolve().parent


def relative_path(*parts):
    return os.path.relpath(SCRIPT_DIR.joinpath(*parts), start=Path.cwd())


# --------------------------------------------------
# Define a differentiable soft mask
# --------------------------------------------------
def soft_mask(H, W, cx, cy, sigma):
    xs = torch.arange(H, device=cx.device).view(H, 1)
    ys = torch.arange(W, device=cx.device).view(1, W)
    mask = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2 + 1e-8))
    return mask  # Return a two-dimensional mask


# --------------------------------------------------
# Initialize adversarial variables
# --------------------------------------------------
def init_adv_soft(image_size, device):
    embedding = torch.randn(
        (1, 3, image_size, image_size),
        device=device,
        requires_grad=True
    )

    cx = (torch.rand(1, device=device) * image_size).detach().requires_grad_(True)
    cy = (torch.rand(1, device=device) * image_size).detach().requires_grad_(True)
    sigma = (torch.rand(1, device=device) * (image_size // 6)).detach().requires_grad_(True)


    return embedding, cx, cy, sigma


# --------------------------------------------------
# Construct adversarial images with a soft mask
# --------------------------------------------------
def construct_embedding_soft(image, embedding, cx, cy, sigma, mean, std):
    B, C, H, W = image.shape

    patch = embedding.repeat(B, 1, 1, 1)

    mask = soft_mask(H, W, cx, cy, sigma)
    mask = mask.unsqueeze(0).unsqueeze(0)        # Add batch and channel dimensions
    mask = mask.repeat(B, C, 1, 1)               # Expand across the batch and channels

    adv = torch.clamp(image + patch * mask, 0, 1)
    adv = (adv - mean) / std
    return adv, mask

def parse_args():
    parser = argparse.ArgumentParser(
                    prog='model-level-backdoor-detection')
    parser.add_argument('--model_name', 
                        default=relative_path('..', 'Models', 'Backdoor_models', 'CLIP-L-336', 'badencoder', 'poisoned_model_0', 'pytorch_model.bin'))
    parser.add_argument('--base_model_name',
                        default=relative_path('..', '..', '..', '..', 'Models', 'clip-vit-large-patch14-336'),
                        help='path to the base vision encoder')
    parser.add_argument('--seed', 
                        default=42,
                        type=int)
    parser.add_argument('--output_folder',
                        default=relative_path('..', 'Results', 'BadEncoder', 'CLIP-L-336', 'poisoned_model_0'))
    parser.add_argument('--device',
                        help='device used for execution',
                        type=str,
                        required=False,
                        default='cuda:7')
    parser.add_argument('--dataset_path',
                        help='path to the dataset',
                        type=str,
                        required=False,
                        default=relative_path('..', 'Data', 'Train_data'))
    parser.add_argument('--image_size',
                        help='image size',
                        type=int,
                        required=False,
                        default=336)
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float,
                        required=False,
                        default=8e-2)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        type=float,
                        required=False,
                        default=1e-5)
    parser.add_argument('--beta',
                        help='beta coefficient in the loss function',
                        type=float,
                        required=False,
                        default=2)
    parser.add_argument('--gamma',
                        help='gamma coefficient in the loss function',
                        type=float,
                        required=False,
                        default=0.01)
    parser.add_argument('--threshold_same_target',
                        help='same-target decision threshold',
                        type=float,
                        required=False,
                        default=0.949)
    parser.add_argument('--epochs',
                        help='number of training epochs',
                        type=int,
                        required=False,
                        default=1)
    parser.add_argument('--backdoored_model',
                        help='whether the model is backdoored',
                        type=str2bool,
                        required=False,
                        default="True")
    parser.add_argument('--scale',
                        help='number of training samples',
                        type=int,
                        required=False,
                        default=5000)
    
    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    else:
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

        
# --------------------------------------------------
# Main program
# --------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device)

    set_random_seed(args.seed)
    os.makedirs(args.output_folder, exist_ok=True)

    print("==== Soft-Mask Backdoor Detection Start ====")

    # Load the model and image processor
    model = CLIPVisionModel.from_pretrained(args.base_model_name)
    processor = AutoProcessor.from_pretrained(args.base_model_name)

    state_dict = torch.load(args.model_name, map_location="cpu")
    
    if 'PBCL' in args.model_name:
        state_dict.pop("visual_projection.weight", None)
        state_dict.pop("visual_projection.bias", None)
        model.load_state_dict(state_dict)
        
    elif "Benign" in args.model_name:
        model.vision_model.load_state_dict(state_dict)
   
    else:
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=device).view(1, 3, 1, 1)

    # Load the dataset
    train_set = Feature_Dataset(args.dataset_path, processor,
                                sample_scale=args.scale, mode='train',image_size=args.image_size)
    test_set = Feature_Dataset(args.dataset_path, processor,
                               sample_scale=args.scale, mode='test',image_size=args.image_size)

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Initialize adversarial parameters
    embedding, cx, cy, sigma = init_adv_soft(args.image_size, device)

    optimizer = torch.optim.Adam(
        [embedding, cx, cy, sigma],
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    loss_a_fn = AssimilationLoss()
    loss_s_fn = SimilarityLoss()

    flag = False
    same_target = 0.0
    begin_time = time.time()
    
    same_target_best = -100000

    Loss_log = []
    
    # --------------------------------------------------
    # Train the adversarial parameters
    # --------------------------------------------------
    for epoch in range(args.epochs):
        for i, image in enumerate(train_loader):
            image = image.to(device)

            clean = (image - mean) / std
            adv, mask = construct_embedding_soft(
                image, embedding, cx, cy, sigma, mean, std
            )

            with torch.no_grad():
                feat_clean = model(clean)[0]

            feat_adv = model(adv)[0]

            loss_a = loss_a_fn(feat_adv)
            loss_s = loss_s_fn(feat_clean, feat_adv)
            loss = -loss_a + args.beta * loss_s + args.gamma * sigma
            
            Loss_log.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Constrain the parameter ranges
            with torch.no_grad():
                cx.clamp_(0, args.image_size - 1)
                cy.clamp_(0, args.image_size - 1)
                sigma.clamp_(1.0, args.image_size / 3)

            if i % 2 == 0:
                print(f"[E{epoch} B{i}] Loss:{loss.item():.4f}  A:{loss_a.item():.4f}, S:{loss_s.item():.4f},   Sigma:{sigma.item():.4f},  CX:{cx.item():.4f},  CY:{cy.item():.4f}")

                if not flag:
                    ratio = attack_success_rate_soft(
                        model, embedding, cx, cy, sigma, device, test_loader, mean, std
                    )
                    print(f"Attack Success Rate: {ratio:.4f}")
                    if ratio > 0.8:
                        same_target = same_target_judge_soft(
                            model, embedding, cx, cy, sigma, device, test_loader, mean, std
                        )
                        print(f"Same Target Ratio: {same_target:.4f}")
                        if same_target > args.threshold_same_target:
                            print(">>> Backdoor Embedding Found!")
                            flag = True
                            break
                        if same_target > 0.9 and same_target < same_target_best:
                            print(">>> Backdoor Embedding Found!")
                            flag = True
                            break
                        same_target_best = same_target
            if sigma.item() < 35:
                break
        if flag:
            break
        
    

    # --------------------------------------------------
    # Save the results
    # --------------------------------------------------
    final_mask = soft_mask(
        args.image_size,
        args.image_size,
        cx.detach(),
        cy.detach(),
        sigma.detach()
    )
    
    result_params = {
        "cx": cx.detach().cpu(),
        "cy": cy.detach().cpu(),
        "sigma": sigma.detach().cpu(),
    }

    torch.save(result_params, f"{args.output_folder}/params.pt")

    torch.save(embedding.detach().cpu(),
               f"{args.output_folder}/Backdoor_Embedding.pt")
    torch.save(final_mask.cpu(),
               f"{args.output_folder}/Soft_Mask.pt")
    feat_adv = model(adv).last_hidden_state.mean(dim=1)
    torch.save(feat_adv[:1].detach().cpu(),
               f"{args.output_folder}/Backdoor_Feature.pt")
    
    with open(f"{args.output_folder}/loss.txt", "w") as f:
        for loss in Loss_log:
            f.write(f"{loss:.4f}\n")

    with open(f"{args.output_folder}/log.txt", "w") as f:
        f.write(f"Backdoored Model: {args.backdoored_model}\n")
        f.write(f"Detected: {flag}\n")
        f.write(f"Same Target Ratio: {same_target:.4f}\n")
        f.write(f"Loss_A: {loss_a.item():.4f}, Loss_S: {loss_s.item():.4f}\n")
        f.write(f"CX: {cx.item():.4f}, CY: {cy.item():.4f}, Sigma: {sigma.item():.4f}\n")
        f.write(f"Time: {time.time() - begin_time:.2f}s\n")

    if not flag:
        print(">>> No Backdoor Detected.")


if __name__ == "__main__":
    main()
