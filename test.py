import torch
import numpy as np
from tqdm import tqdm
from Utils.loss import AssimilationLoss,SimilarityLoss

def construct_embedding(original_tensor, new_tensor, mean, std):
    new_tensor = torch.repeat_interleave(
                    new_tensor,
                    original_tensor.shape[0],
                    dim=0)
    
    modified_tensor = torch.clamp(original_tensor + new_tensor,0,1)
    
    modified_tensor = (modified_tensor - mean) / std
    
    return modified_tensor

def attack_success_rate(custom_encoder, embedding, device, test_loader, mean, std):
    loss_assimilation = AssimilationLoss()
    num_success = 0
    with torch.no_grad():
        for i, image in enumerate(test_loader):
            image = image.to(device)
            new_embedding = construct_embedding(image, embedding, mean, std) 
            
            # forward pass
            new_feature = custom_encoder(new_embedding)[0]

            loss_a = loss_assimilation(new_feature)
            
            if loss_a.item() > 0.6:
                num_success += 1
    
    ratio = num_success / len(test_loader)
    
    
    
    return ratio


def same_target_judge(custom_encoder, embedding, device, test_loader, mean, std):
    '''Judge if all the features in the batch are the same one'''
    loss_assimilation = SimilarityLoss()
    num_success = 0
    with torch.no_grad():
        for i, image in enumerate(test_loader):
            image = image.to(device)
            new_embedding = construct_embedding(image, embedding, mean, std) 
            new_embedding = new_embedding.to(device)

            # forward pass
            new_feature = custom_encoder(new_embedding)[0]
            
            if i == 0:
                old_feature = new_feature
            else:
                loss_s = loss_assimilation(new_feature, old_feature)
                if loss_s.item() > 0.8:
                    num_success += 1
    
    ratio = num_success / len(test_loader)
    
    return ratio

def soft_mask(H, W, cx, cy, sigma):
    xs = torch.arange(H, device=cx.device).view(H, 1)
    ys = torch.arange(W, device=cx.device).view(1, W)
    mask = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2 + 1e-8))
    return mask  # [H, W]

def construct_embedding_soft(image, embedding, cx, cy, sigma, mean, std):
    B, C, H, W = image.shape

    patch = embedding.repeat(B, 1, 1, 1)

    mask = soft_mask(H, W, cx, cy, sigma)
    mask = mask.unsqueeze(0).unsqueeze(0)        # [1,1,H,W]
    mask = mask.repeat(B, C, 1, 1)                # [B,C,H,W]

    adv = torch.clamp(image + patch * mask, 0, 1)
    adv = (adv - mean) / std
    return adv, mask

def attack_success_rate_soft(custom_encoder, embedding, cx, cy, sigma, device, test_loader, mean, std):
    loss_assimilation = AssimilationLoss()
    num_success = 0
    with torch.no_grad():
        for i, image in enumerate(test_loader):
            image = image.to(device)
            new_embedding, mask = construct_embedding_soft(
                image, embedding, cx, cy, sigma, mean, std
            )
            # forward pass
            new_feature = custom_encoder(new_embedding)[0]

            loss_a = loss_assimilation(new_feature)
            
            if loss_a.item() > 0.6:
                num_success += 1
    
    ratio = num_success / len(test_loader)
    
    
    
    return ratio


def same_target_judge_soft(custom_encoder, embedding, cx, cy, sigma, device, test_loader, mean, std):
    '''Judge if all the features in the batch are the same one'''
    loss_assimilation = SimilarityLoss()
    num_success = 0
    with torch.no_grad():
        for i, image in enumerate(test_loader):
            image = image.to(device)
            new_embedding, mask = construct_embedding_soft(
                image, embedding, cx, cy, sigma, mean, std
            )
            new_embedding = new_embedding.to(device)

            # forward pass
            new_feature = custom_encoder(new_embedding)[0]
            
            if i == 0:
                old_feature = new_feature
            else:
                loss_s = loss_assimilation(new_feature, old_feature)
                if loss_s.item() > 0.8:
                    num_success += 1
    
    ratio = num_success / len(test_loader)
    
    return ratio

def same_target_judge_soft_open_clip(custom_encoder, embedding, cx, cy, sigma, device, test_loader, mean, std):
    '''Judge if all the features in the batch are the same one'''
    loss_assimilation = SimilarityLoss()
    num_success = 0
    with torch.no_grad():
        for i, image in enumerate(test_loader):
            image = image.to(device)
            new_embedding, mask = construct_embedding_soft(
                image, embedding, cx, cy, sigma, mean, std
            )
            new_embedding = new_embedding.to(device)

            # forward pass
            _, new_feature = custom_encoder.encode_image(new_embedding, normalize=False)
            
            if i == 0:
                old_feature = new_feature
            else:
                loss_s = loss_assimilation(new_feature, old_feature)
                if loss_s.item() > 0.8:
                    num_success += 1
    
    ratio = num_success / len(test_loader)
    
    return ratio
        
def attack_success_rate_soft_open_clip(custom_encoder, embedding, cx, cy, sigma, device, test_loader, mean, std):
    loss_assimilation = AssimilationLoss()
    num_success = 0
    with torch.no_grad():
        for i, image in enumerate(test_loader):
            image = image.to(device)
            new_embedding, mask = construct_embedding_soft(
                image, embedding, cx, cy, sigma, mean, std
            )
            # forward pass
            _, new_feature = custom_encoder.encode_image(new_embedding, normalize=False)

            loss_a = loss_assimilation(new_feature)
            
            if loss_a.item() > 0.6:
                num_success += 1
    
    ratio = num_success / len(test_loader)
    
    
    
    return ratio